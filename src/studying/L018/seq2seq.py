import os
import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import bokeh.plotting as bp
import bokeh.models as bm

from nltk.tokenize import WordPunctTokenizer
from nltk.translate.bleu_score import corpus_bleu
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from tqdm import tqdm, trange
from bokeh.io import output_file, show
from vocab import Vocab

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BasicModel(nn.Module):
    def __init__(self, inp_voc, out_voc, emb_size=64, hid_size=128):
        super().__init__()
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hid_size = hid_size
        self.emb_size = emb_size

        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True)

        self.dec_start = nn.Linear(hid_size, hid_size)
        self.dec0 = nn.GRUCell(emb_size, hid_size)
        self.logits = nn.Linear(hid_size, len(out_voc))

    def forward(self, inp, out):
        initial_state = self.encode(inp)
        return self.decode(initial_state, out)

    def encode(self, inp, **flags):
        """
        Принимает на вход символьную последовательность, вычисляет начальное состояние
        :param inp: матрица входных токенов [batch, time]
        :returns: тензоры начального состояние декодера (один или несколько)
        """
        inp_emb = self.emb_inp(inp)
        batch_size = inp.shape[0]

        # enc_seq: [batch, time, hid_size], last_state: [batch, hid_size]
        enc_seq, [last_state_but_not_really] = self.enc0(inp_emb)

        # замечание: last_state на самом деле не последнее состояние, так как в последовательностях используется padding
        # выведем реальный last_state
        lengths = (inp != self.inp_voc.eos_ix).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]
        # ^-- shape: [batch_size, hid_size]

        dec_state = self.dec_start(last_state)
        return [dec_state]

    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Принимает предыдущее состояние декодера и токены, возвращает новое состояние и логиты для следующих токенов
        :param prev_state: массив (list) тензоров предыдущих состояний декодера, аналогичный возвращаемому encoder(...)
        :param prev_tokens: выходные токены предыдущей итерации, целочисленный вектор [batch_size]
        :return: массив (list) тензоров следующих состояний декодера, тензор с логитами [batch, len(out_voc)]
        """
        prev_gru0_state = prev_state[0]

        prev_emb = self.emb_out(prev_tokens)
        new_dec_state = self.dec0(prev_emb, prev_gru0_state)

        output_logits = torch_f.softmax(self.logits(new_dec_state), dim=1)
        output_logits = torch.log(output_logits + 1e-9)

        return [new_dec_state], output_logits

    def decode(self, initial_state, out_tokens, **flags):
        """ Итерирование по референсным токенам (выходным) с шагом декодера """
        batch_size = out_tokens.shape[0]
        state = initial_state

        # начальные логиты: всегда предсказывают токен BOS
        onehot_bos = torch_f.one_hot(
            torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64),
            num_classes=len(self.out_voc)
        ).to(device=out_tokens.device)
        first_logits = torch.log(onehot_bos.to(torch.float32) + 1e-9)

        logits_sequence = [first_logits]
        for i in range(out_tokens.shape[1] - 1):
            state, logits = self.decode_step(state, out_tokens[:, i])
            logits_sequence.append(logits)
        return torch.stack(logits_sequence, dim=1)

    def decode_inference(self, initial_state, max_len=100, **flags):
        """ Генерация переводов моделью (жадная версия алгоритма) """
        batch_size, device = len(initial_state[0]), initial_state[0].device
        state = initial_state
        outputs = [torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64, device=device)]
        all_states = [initial_state]

        for i in range(max_len):
            state, logits = self.decode_step(state, outputs[-1])
            outputs.append(logits.argmax(dim=-1))
            all_states.append(state)

        return torch.stack(outputs, dim=1), all_states

    def translate_lines(self, inp_lines, **kwargs):
        inp = self.inp_voc.to_matrix(inp_lines).to(device)
        initial_state = self.encode(inp)
        out_ids, states = self.decode_inference(initial_state, **kwargs)
        return self.out_voc.to_lines(out_ids.cpu().numpy()), states


class AttentionLayer(nn.Module):
    def __init__(self, name, enc_size, dec_size, hid_size, activ=torch.tanh):
        """ Слой, подсчитывающий выходы аттеншена и веса """
        super().__init__()
        self.name = name
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.hid_size = hid_size
        self.activ = activ

        # создаем обучаемые параметры:
        self.w_e = nn.parameter.Parameter(torch.rand(enc_size, hid_size), requires_grad=True)
        self.w_d = nn.parameter.Parameter(torch.rand(dec_size, hid_size), requires_grad=True)
        self.w_out = nn.parameter.Parameter(torch.rand(hid_size, 1), requires_grad=True)

    def forward(self, enc, dec, inp_mask):
        """
        Подсчитывает выход аттеншена и веса
        :param enc: входная последовательность кодировщика, float32[batch_size, ninp, enc_size]
        :param dec: выходная последовательность декодировщика (query), float32[batch_size, dec_size]
        :param inp_mask: маска для последовательностей кодировщика (0 после первого токена eos), float32 [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
            - attn - вектор выхода аттеншена (взвешенная сумма enc)
            - probs - веса аттеншена после софтмакса (softmax)
        """
        batch_size = enc.shape[0]

        # так как вектор декодировщика один, а входных много, добавим размерность в тензор для бродкастинга
        dec = dec[:, None, :]

        # Подсчет логитов
        a_t = self.activ(enc @ self.w_e + dec @ self.w_d) @ self.w_out

        # Наложение маски - если mask равна 0, логиты должны быть -inf или -1e9
        # Вам может понадобиться torch.where
        a_t[~inp_mask] = -torch.inf
        a_t = a_t.squeeze(dim=-1)

        # Подсчет вероятностей аттеншена (softmax)
        probs = torch_f.softmax(a_t, dim=-1)

        # Подсчет выхода аттеншена, используя enc и probs
        attn = (probs[:, :, None] * enc).sum(dim=1)

        return attn, probs


class AttentiveModel(BasicModel):
    def __init__(self, name, inp_voc, out_voc, emb_size=64, hid_size=128, attn_size=32):
        """ Модель машинного перевода с использованием механизма внимания (описание модели дано выше) """
        super().__init__(inp_voc, out_voc, emb_size, hid_size)

        self.attn = AttentionLayer('attn_layer', hid_size, hid_size, attn_size)
        self.dec0 = nn.GRUCell(emb_size + hid_size, hid_size)

    def encode(self, inp, **flags):
        """
        Принимает на вход символьную последовательность, считает начальное состояние
        :param inp: матрица входных токенов [batch, time]
        :return: массив тензоров начальных состояний декодера
        """

        # кодирование входной последовательности и создание начального состояния для декодеровщика
        inp_emb = self.emb_inp(inp)
        enc_seq, [last_state_but_not_really] = self.enc0(inp_emb)
        lengths = (inp != self.inp_voc.eos_ix).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]
        dec_start = self.dec_start(last_state)
        zero_start = torch.zeros_like(dec_start)

        # применение слоя аттеншен к начальному состоянию декодеровщика
        inp_mask = self.inp_voc.compute_mask(inp)
        first_attn, attn_probs = self.attn(enc_seq, zero_start, inp_mask)

        # Создаем первый тензор состояния, включающий:
        # * начальные состояния для декодеровщика
        # * закодированную последовательность и аттеншен маску для нее
        # * последним элементом тензора должны идти вероятности из аттеншена

        first_state = [zero_start, enc_seq, inp_mask, attn_probs]
        return first_state

    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Принимает на вход предыдущее состояние декодеровщика, возвращает новое его состояние и логиты для следующих токенов
        :param prev_state: массив тензоров предыдущих состояний декодера
        :param prev_tokens: предыдущие токены из выхода слоя, целочисленный вектор [batch_size]
        :return: массив тензоров следующего состояния декодера, тензор логитов [batch, n_tokens]
        """

        dec_hidden, enc_seq, inp_mask, _ = prev_state
        attn_dec, attn_probs = self.attn(enc_seq, dec_hidden, inp_mask)

        prev_emb = self.emb_out(prev_tokens).view(-1, self.emb_size)

        dec_in = torch.concat([prev_emb, attn_dec], dim=-1)
        new_dec_hidden = self.dec0(dec_in, dec_hidden)

        output_logits = torch_f.softmax(self.logits(new_dec_hidden), dim=-1)
        output_logits = torch.log(output_logits)
        new_dec_state = [new_dec_hidden, enc_seq, inp_mask, attn_probs]

        return new_dec_state, output_logits


class AttentiveBSModel(AttentiveModel):
    def __init__(self, name, inp_voc, out_voc, n_beams, emb_size=64, hid_size=128, attn_size=32):
        # super().__init__(name, inp_voc, out_voc)
        super().__init__(name, inp_voc, out_voc, emb_size, hid_size, attn_size)
        self.n_beams = n_beams

    def decode_inference(self, initial_state, max_len=100, **flags):
        """ Генерация переводов моделью beam search """
        batch_size, device = len(initial_state[0]), initial_state[0].device

        # будем сохранять стейты, id токенов на выходе модели и накопленное значение логитов последовательности

        # заполнение начала строки
        state = initial_state
        outputs = [torch.full([batch_size, self.n_beams], self.out_voc.bos_ix, dtype=torch.int64, device=device)]
        probs = torch.full([batch_size, self.n_beams], 0, dtype=torch.float64, device=device)

        # заполнение первого слова
        first_state, first_output = self.decode_step(initial_state, outputs[-1][:, 0])
        most_prob_outputs = first_output.argsort(dim=-1, descending=True)[:, :self.n_beams]
        assert most_prob_outputs.shape == (batch_size, self.n_beams)

        prev_outputs = [outputs[-1]]
        outputs.append(most_prob_outputs)
        probs += torch.take_along_dim(first_output, most_prob_outputs, dim=-1)
        assert probs.shape == (batch_size, self.n_beams)

        all_states = [initial_state, [first_state for _ in range(self.n_beams)]]
        states_nums = [torch.full([batch_size, self.n_beams],0, dtype=torch.int64, device=device)]

        # заполнение остальных слов
        for i in range(1, max_len):
            outputs_bm = []
            probs_bm = []
            states_bm = []
            prev_outputs_bm = []

            for bm in range(self.n_beams):
                prev_output = outputs[-1][:, bm].view(-1, 1)
                assert prev_output.shape == (batch_size, 1)

                # prev_state у каждого объекта свой, нужно пройтись циклом по всем n_beams возможных
                # prev_state и записать их выходы согласно индексам
                prev_state_list = all_states[-1]
                # стейты, которые будут заполняться в цикле ниже
                prev_state = [torch.zeros_like(elem) for elem in first_state]

                for i_pst in range(self.n_beams):
                    idx = (states_nums[-1][:, bm] == i_pst)
                    if idx.sum() > 0:
                        for n_elem in range(len(prev_state)):
                            prev_state[n_elem][idx] = prev_state_list[i_pst][n_elem][idx]

                # для каждого bm получаем новый state и логиты
                state_bm, logits_bm = self.decode_step(prev_state, prev_output)
                states_bm.append(state_bm)

                # берем наиболее вероятные n_beams логитов
                most_prob_outputs = logits_bm.argsort(dim=-1, descending=True)[:, :self.n_beams]
                assert most_prob_outputs.shape == (batch_size, self.n_beams)

                outputs_bm.append(most_prob_outputs)

                # вычисляем новый логит (log(p_t * p_(t-1)) = log(p_t) + log(p_(t-1)))
                assert probs[:, bm].view(-1, 1).shape == (batch_size, 1)
                curr_probs = probs[:, bm].view(-1, 1) + torch.take_along_dim(logits_bm, most_prob_outputs, dim=-1)

                assert curr_probs.shape == (batch_size, self.n_beams)
                probs_bm.append(curr_probs)

                prev_outputs_bm.append(prev_output.view(-1, 1).expand(-1, self.n_beams))
                assert prev_outputs_bm[-1].shape == (batch_size, self.n_beams)

            assert len(outputs_bm) == self.n_beams
            assert len(probs_bm) == self.n_beams
            assert len(states_bm) == self.n_beams

            # приведем листы outputs и probs к размеру [batch_size, n_beams ** 2]
            outputs_bm = torch.flatten(torch.stack(outputs_bm, dim=1), start_dim=1)
            probs_bm = torch.flatten(torch.stack(probs_bm, dim=1), start_dim=1)
            prev_outputs_bm = torch.flatten(torch.stack(prev_outputs_bm, dim=1), start_dim=1)
            assert outputs_bm.shape == (batch_size, self.n_beams ** 2)
            assert probs_bm.shape == (batch_size, self.n_beams ** 2)
            assert prev_outputs_bm.shape == (batch_size, self.n_beams ** 2)

            # среди всех n_beams ** 2 логитов находим наиболее вероятные n_beams штук и записываем в outputs
            high_prob_beams = probs_bm.argsort(dim=-1, descending=True)[:, :self.n_beams]
            high_prob_outputs = (torch.take_along_dim(outputs_bm, high_prob_beams, dim=-1))
            outputs.append(high_prob_outputs)
            assert outputs[-1].shape == (batch_size, self.n_beams)
            prev_outputs.append(torch.take_along_dim(prev_outputs_bm, high_prob_beams, dim=-1))

            # обновляем накопленные логиты
            probs = torch.take_along_dim(probs_bm, high_prob_beams, dim=-1)
            assert probs.shape == (batch_size, self.n_beams)

            # выясним, в каких beams были максимальные накопленные логиты
            # и сохраним соответствующие state и prev_outputs
            high_prob_beams_nums = torch.div(high_prob_beams, self.n_beams, rounding_mode='floor')
            assert high_prob_beams_nums.shape == (batch_size, self.n_beams)
            states_nums.append(high_prob_beams_nums)
            all_states.append(states_bm)
            assert len(all_states[-1]) == self.n_beams
            assert len(all_states[-1][0]) == len(initial_state)

        # выбираем последовательность, имеющую наибольшую вероятность
        max_prob = probs.argmax(dim=-1, keepdim=True)
        assert max_prob.shape == (batch_size, 1)

        # разматываем последовательность к началу,
        # сопоставляя наиболее вероятные токены и соответствующие им предыдущие
        most_prob_seq = []
        while len(prev_outputs) > 0:
            curr_output = outputs.pop()
            most_prob_seq = [torch.take_along_dim(curr_output, max_prob, dim=-1)] + most_prob_seq
            curr_prev = prev_outputs.pop()
            prev = torch.take_along_dim(curr_prev, max_prob, dim=-1)
            max_prob = (outputs[-1] == prev).int().argmax(dim=-1, keepdims=True)

        return torch.stack(most_prob_seq, dim=1), all_states

    def translate_lines(self, inp_lines, **kwargs):
        inp = self.inp_voc.to_matrix(inp_lines).to(device)
        initial_state = self.encode(inp)
        out_ids, state = self.decode_inference(initial_state, **kwargs)
        out_lines = self.out_voc.to_lines(out_ids)

        return out_lines, state[-1]


def print_header(h: str):
    print(f'===== {h.upper()} =====')


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run():
    print_header('create tokenizer')
    tokenizer = WordPunctTokenizer()

    def tokenize(x):
        return ' '.join(tokenizer.tokenize(x.lower()))

    # сплит и токенизация данных
    print_header('split & takenization of data')
    if not os.path.isfile('./data/train.en') or not os.path.isfile('./data/train.ru'):
        with open('./data/train.en', 'w', encoding='utf-8') as src_file, open('./data/train.ru', 'w', encoding='utf-8') as dst_file:
            for line in open('./data/data.txt', encoding='utf-8'):
                src_line, dst_line = line.strip().split('\t')
                src_file.write(tokenize(src_line) + '\n')
                dst_file.write(tokenize(dst_line) + '\n')
        print('train files are wrote')

    print_header('create BPE-dict + dict applying to data')
    langs = ['en', 'ru']
    if not os.path.isfile('./data/bpe_rules.en') or not os.path.isfile('./data/bpe_rules.en'):
        for lang in langs:
            learn_bpe(
                open('./data/train.' + lang, encoding='utf-8'),
                open('./data/bpe_rules.' + lang, 'w', encoding='utf-8'),
                num_symbols=8_000
            )
        print('bpe_rules are wrote')

    bpe = {}
    for lang in langs:
        bpe[lang] = BPE(open('./data/bpe_rules.' + lang, encoding='utf-8'))

    if not os.path.isfile('./data/train.bpe.ru') or not os.path.isfile('./data/train.bpe.en'):
        for lang in langs:
            with open('./data/train.bpe.' + lang, 'w', encoding='utf-8') as out_file:
                for line in open('./data/train.' + lang, encoding='utf-8'):
                    out_file.write(bpe[lang].process_line(line.strip()) + '\n')
    print('bre rules are wrote')

    data_inp = np.array(open('./data/train.bpe.ru', encoding='utf-8').read().split('\n'))
    data_out = np.array(open('./data/train.bpe.en', encoding='utf-8').read().split('\n'))

    train_inp, dev_inp, train_out, dev_out = train_test_split(
        data_inp,
        data_out,
        test_size=3000,
        random_state=42
    )

    for i in range(3):
        print(f'inp: {train_inp[i]}')
        print(f'out: {train_out[i]}', end='\n\n')

    inp_voc = Vocab.from_lines(train_inp)
    out_voc = Vocab.from_lines(train_out)

    # Так работает перевод строк в ID и обратно
    batch_lines = sorted(train_inp, key=len)[5:10]
    batch_ids = inp_voc.to_matrix(batch_lines)
    batch_lines_restored = inp_voc.to_lines(batch_ids)
    print('lines')
    print(batch_lines)
    print('\nwords to ids (0 = bos, 1 = eos):')
    print(batch_ids)
    print('\nback to words')
    print(batch_lines_restored)

    plt.figure(figsize=[10, 5])
    plt.subplot(1, 2, 1)
    plt.title('Length before translation')
    plt.hist(list(map(len, map(str.split, train_inp))), bins=20)

    plt.subplot(1, 2, 2)
    plt.title('Length after translation')
    plt.hist(list(map(len, map(str.split, train_out))), bins=20)
    # plt.show()

    set_random_seed(1)

    print_header('basic model without training')
    model = BasicModel(inp_voc, out_voc).to(device)
    dummy_inp_tokens = inp_voc.to_matrix(sorted(train_inp, key=len)[5:10]).to(device)
    dummy_out_tokens = out_voc.to_matrix(sorted(train_out, key=len)[5:10]).to(device)

    h0 = model.encode(dummy_inp_tokens)
    h1, logits1 = model.decode_step(h0, torch.arange(len(dummy_inp_tokens), device=device))

    assert isinstance(h1, list) and len(h1) == len(h0)
    assert h1[0].shape == h0[0].shape and not torch.allclose(h1[0], h0[0])
    assert logits1.shape == (len(dummy_inp_tokens), len(out_voc))

    logits_seq = model.decode(h0, dummy_out_tokens)
    assert logits_seq.shape == (dummy_out_tokens.shape[0], dummy_out_tokens.shape[1], len(out_voc))

    print('полный проход')
    logits_seq2 = model(dummy_inp_tokens, dummy_out_tokens)
    assert logits_seq2.shape == logits_seq.shape

    # print('\n'.join([line for line in train_inp[:3]]))
    dummy_translations, dummy_states = model.translate_lines(train_inp[:3], max_len=25)
    print("Перевод необученной моделью:")
    print('\n'.join([line for line in dummy_translations]))

    def compute_loss(model, inp, out, **flags):
        """
        Считает лосс (float32 scalar) по формуле выше
        :param inp: входная матрица токенов, int32[batch, time]
        :param out: референсная (таргет) матрица токенов, int32[batch, time]

        Чтобы пройти тесты, ваша функция должна
        * включать в подсчет лосса первое вхождение токена EOS, но не последующие для одной последовательности
        * делить сумму лоссов на сумму длин входных последовательностей (используйте voc.compute_mask)
        """
        # compute a boolean mask that equals "1" until first EOS (including that EOS)
        mask = model.out_voc.compute_mask(out)  # [batch_size, out_len]
        targets_1hot = torch_f.one_hot(out, len(model.out_voc)).to(torch.float32)  # [batch_size, out_len, num_tokens]

        # выходы модели, [batch_size, out_len, num_tokens]
        logits_seq = model(inp, out)

        # логарифмы вероятностей для всех токенов на всех шагах, [batch_size, out_len, num_tokens]
        # вычислены на предыдущем шаге - выходы модели - логарифмы

        # логарифмы вероятностей правильных выходов модели, [batch_size, out_len]
        logp_out = (logits_seq * targets_1hot).sum(dim=-1)
        # ^-- этот код выбирает вероятности для реального следующего токена (следующий токен из таргета)
        # Замечание: более эффективно можно посчитать лосс, используя F.cross_entropy

        # усредненная кросс-энтропия по токенам, где mask == True
        return -logp_out[mask].mean()  # усредненный лосс, scalar

    dummy_loss = compute_loss(model, dummy_inp_tokens, dummy_out_tokens)
    print("Loss:", dummy_loss)
    assert np.allclose(dummy_loss.item(), 7.5, rtol=0.1, atol=0.1), "We're sorry for your loss"

    # test autograd
    dummy_loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and abs(param.grad.max()) != 0, f"Param {name} received no gradients"

    def compute_bleu(model, inp_lines, out_lines, bpe_sep='@@ ', batch_size=512, **flags):
        """
        Вычисляет BLEU скор модели (на уровне корпусов текстов) по входным данным и референсному переводу
        Замечание: для более аккуратного подсчета BLEU и сравнения результатов стоит использовать https://pypi.org/project/sacrebleu
        """
        n_batches = math.ceil(len(inp_lines) / batch_size)

        with torch.no_grad():
            translations = []

            for i in range(n_batches):
                inp_lines_i = inp_lines[i * batch_size: (i + 1) * batch_size]
                translations_i, _ = model.translate_lines(inp_lines_i, **flags)
                translations_i = [line.replace(bpe_sep, '') for line in translations_i]
                translations.extend(translations_i)

            actual = [line.replace(bpe_sep, '') for line in out_lines]
            return corpus_bleu(
                [[ref.split()] for ref in actual],
                [trans.split() for trans in translations],
                smoothing_function=lambda precisions, **kw: [p + 1.0 / p.denominator for p in precisions]
            ) * 100

    print(compute_bleu(model, dev_inp, dev_out))

    print_header('training')
    metrics = {'train_loss': [], 'dev_bleu': []}

    model = BasicModel(inp_voc, out_voc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32

    step_amount = 25_000
    for _ in trange(step_amount):
        step = len(metrics['train_loss']) + 1
        batch_ix = np.random.randint(len(train_inp), size=batch_size)
        batch_inp = inp_voc.to_matrix(train_inp[batch_ix]).to(device)
        batch_out = out_voc.to_matrix(train_out[batch_ix]).to(device)

        loss_t = compute_loss(model, batch_inp, batch_out)
        loss_t.backward()
        opt.step()
        opt.zero_grad()

        metrics['train_loss'].append((step, loss_t.item()))

        if step % 1_000 == 0:
            metrics['dev_bleu'].append((step, compute_bleu(model, dev_inp, dev_out)))

            clear_output(True)
            if step == 25_000:
                plt.figure(figsize=(12, 4))
                for i, (name, history) in enumerate(sorted(metrics.items())):
                    plt.subplot(1, len(metrics), i + 1)
                    plt.title(name)
                    plt.plot(*zip(*history))
                    plt.grid()
                plt.show()
            print("Mean loss=%.3f" % np.mean(metrics['train_loss'][-10:], axis=0)[1], flush=True)
    # Замечание: колебания BLEU на графике вполне нормальны до тех пор пока в среднем в долгосрочной перспективе (например, 5к шагов или батчей) есть положительные изменения

    assert np.mean(metrics['dev_bleu'][-10:], axis=0)[1] > 15, "Нужно больше BLEU, еще больше"

    for inp_line, trans_line in zip(dev_inp[::500], model.translate_lines(dev_inp[::500])[0]):
        print(inp_line)
        print(trans_line)
        print()



# ---------------------------
# set_random_seed(1)
# model_attn = AttentiveModel('attentive', inp_voc, out_voc).to(device)
# dummy_translations, dummy_states = model_attn.translate_lines(train_inp[:3], max_len=25)
# print("Перевод необученной моделью:")
# print('\n'.join([line for line in dummy_translations]))


# -----------------------------
# metrics = {'train_loss': [], 'dev_bleu': []}
#
# lr = 1e-3
# batch_size = 32
#
# opt = torch.optim.Adam(model_attn.parameters(), lr=lr)
#
# for _ in trange(25000):
#     model_attn.train()
#     step = len(metrics['train_loss']) + 1
#     batch_ix = np.random.randint(len(train_inp), size=batch_size)
#     batch_inp = inp_voc.to_matrix(train_inp[batch_ix]).to(device)
#     batch_out = out_voc.to_matrix(train_out[batch_ix]).to(device)
#
#     loss_t = compute_loss(model_attn, batch_inp, batch_out)
#     loss_t.backward()
#     opt.step()
#     opt.zero_grad()
#
#     metrics['train_loss'].append((step, loss_t.item()))
#
#     if step % 100 == 0:
#         with torch.no_grad():
#             metrics['dev_bleu'].append((step, compute_bleu(model_attn, dev_inp, dev_out)))
#
#             clear_output(True)
#             plt.figure(figsize=(12, 4))
#             for i, (name, history) in enumerate(sorted(metrics.items())):
#                 plt.subplot(1, len(metrics), i + 1)
#                 plt.title(name)
#                 plt.plot(*zip(*history))
#                 plt.grid()
#             plt.show()
#         print("Mean loss=%.3f" % np.mean(metrics['train_loss'][-10:], axis=0)[1], flush=True)




# -------------------
# for inp_line, trans_line in zip(dev_inp[::500], model_attn.translate_lines(dev_inp[::500])[0]):
#     print(inp_line)
#     print(trans_line)
#     print()



# -------
# compute_bleu(model_attn, dev_inp, dev_out)

# ------------

#
# output_notebook()
#
#
# def draw_attention(inp_line, translation, probs):
#     """ Функция для визуализации весов аттеншена """
#     inp_tokens = inp_voc.tokenize(inp_line)
#     trans_tokens = out_voc.tokenize(translation)
#     probs = probs[:len(trans_tokens), :len(inp_tokens)]
#
#     fig = pl.figure(x_range=(0, len(inp_tokens)), y_range=(0, len(trans_tokens)),
#                     x_axis_type=None, y_axis_type=None, tools=[])
#     fig.image([probs[::-1]], 0, 0, len(inp_tokens), len(trans_tokens))
#
#     fig.add_layout(bm.LinearAxis(axis_label='source tokens'), 'above')
#     fig.xaxis.ticker = np.arange(len(inp_tokens)) + 0.5
#     fig.xaxis.major_label_overrides = dict(zip(np.arange(len(inp_tokens)) + 0.5, inp_tokens))
#     fig.xaxis.major_label_orientation = 45
#
#     fig.add_layout(bm.LinearAxis(axis_label='translation tokens'), 'left')
#     fig.yaxis.ticker = np.arange(len(trans_tokens)) + 0.5
#     fig.yaxis.major_label_overrides = dict(zip(np.arange(len(trans_tokens)) + 0.5, trans_tokens[::-1]))
#
#     show(fig)







# --------
# inp = dev_inp[::500]
#
# trans, states = model_attn.translate_lines(inp)
#
# # возьмите тензор вероятностей аттеншена из тензора состояния модели (в вашей кастомной модели порядок и конфигурация могу быть другими)
# # тензор вероятностей должен иметь форму[batch_size, translation_length, input_length]
# # например, если эти вероятности находятся в конце каждого тензора состояния, можно использовать np.stack([state[-1] for state in states], axis=1)
# attention_probs = np.stack([state[-1].cpu().detach().numpy() for state in states], axis=1)




# ------------------------
# for i in range(5):
#     draw_attention(inp[i], trans[i], attention_probs[i])
#
# # Выглядит уже лучше? Не забудьте сохранить картинки для сабмита в anytask



# -------------------------
# set_random_seed(1)
# model_attn_bs = AttentiveBSModel('attentive_bs', inp_voc, out_voc, n_beams=4).to(device)
# dummy_translations, dummy_states = model_attn_bs.translate_lines(train_inp[:3], max_len=25)
# print("Перевод необученной моделью:")
# print('\n'.join([line for line in dummy_translations[:3]]))



# ------------------------
# set_random_seed(1)
# model_attn_bs = AttentiveBSModel('attentive_bs', inp_voc, out_voc, n_beams=4).to(device)
# metrics = {'train_loss': [], 'dev_bleu': []}
#
# lr = 1e-3
# batch_size = 32
#
# opt = torch.optim.Adam(model_attn_bs.parameters(), lr=lr)
#
# for _ in trange(25000):
#     model_attn_bs.train()
#     step = len(metrics['train_loss']) + 1
#     batch_ix = np.random.randint(len(train_inp), size=batch_size)
#     batch_inp = inp_voc.to_matrix(train_inp[batch_ix]).to(device)
#     batch_out = out_voc.to_matrix(train_out[batch_ix]).to(device)
#
#     loss_t = compute_loss(model_attn_bs, batch_inp, batch_out)
#     loss_t.backward()
#     opt.step()
#     opt.zero_grad()
#
#     metrics['train_loss'].append((step, loss_t.item()))
#
#     if step % 500 == 0:
#         with torch.no_grad():
#             metrics['dev_bleu'].append((step, compute_bleu(model_attn_bs, dev_inp, dev_out,
#                                                            batch_size=64)))
#
#             clear_output(True)
#             plt.figure(figsize=(12, 4))
#             for i, (name, history) in enumerate(sorted(metrics.items())):
#                 plt.subplot(1, len(metrics), i + 1)
#                 plt.title(name)
#                 plt.plot(*zip(*history))
#                 plt.grid()
#             plt.show()
#         print("Mean loss=%.3f" % np.mean(metrics['train_loss'][-10:], axis=0)[1], flush=True)



# --------------------------------
# print("bleu", metrics["dev_bleu"][-1])


# -------------------------
# for inp_line, trans_line in zip(dev_inp[::500], model_attn_bs.translate_lines(dev_inp[::500])[0]):
#     print(inp_line)
#     print(trans_line)
#     print()


if __name__ == '__main__':
    run()
