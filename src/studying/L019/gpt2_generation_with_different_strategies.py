import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def print_header(h: str):
    print(f'===== {h.upper()} =====')


def run():

    print_header('create tokenizer & model')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # add the EOS token as PAD token to avoid warnings
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

    print_header('encode context the generation is conditioned on')
    input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')
    print(f'input_ids: {input_ids}')

    # print_header('generate text until the output length (which includes the context length) reaches 50')
    # greedy_output = model.generate(input_ids, max_length=50)
    # print(f'greedy_output: {greedy_output}')
    # print('Greedy output:')
    # print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
    #
    # print_header('activate beam search and early_stopping')
    # beam_output = model.generate(
    #     input_ids,
    #     max_length=50,
    #     num_beams=5,
    #     early_stopping=True
    # )
    # print(f'beam_output: {beam_output}')
    # print('Output:')
    # print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
    #
    # print_header('set no_repeat_ngram_size to 2')
    # beam_output = model.generate(
    #     input_ids,
    #     max_length=50,
    #     num_beams=5,
    #     no_repeat_ngram_size=2,
    #     early_stopping=True
    # )
    # print(f'beam_output: {beam_output}')
    # print('Output:')
    # print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
    #
    # print_header('set return_num_sequences > 1')
    # beam_outputs = model.generate(
    #     input_ids,
    #     max_length=50,
    #     num_beams=5,
    #     no_repeat_ngram_size=2,
    #     num_return_sequences=5,
    #     early_stopping=True
    # )
    # print(f'beam_out: {beam_outputs}')
    # for i, beam_output in enumerate(beam_outputs):
    #     print(f'{i}) {tokenizer.decode(beam_output,skip_special_tokens=True)}')
    #
    # print_header('torch.random.manual_seed(1)')
    # torch.random.manual_seed(1)
    #
    # print_header('activate sampling and deactivate top_k by setting top_k sampling to 0')
    # sample_output = model.generate(
    #     input_ids,
    #     do_sample=True,
    #     max_length=50,
    #     top_k=0
    # )
    # print(f'sample_out: {sample_output}')
    # print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
    #
    # print_header('torch.random.manual_seed(0)')
    # torch.random.manual_seed(0)
    # print_header('use temperature to decrease the sensitivity to low probability candidates')
    # sample_output = model.generate(
    #     input_ids,
    #     do_sample=True,
    #     max_length=50,
    #     top_k=0,
    #     temperature=0.7
    # )
    # print(f'sample_out: {sample_output}')
    # print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
    #
    # print_header('set seed to reproduce results. Feel free to change the seed though to get different results')
    # print('torch.random.manual_seed(0)')
    # torch.random.manual_seed(0)
    # sample_output = model.generate(
    #     input_ids,
    #     do_sample=True,
    #     max_length=50,
    #     top_p=0.9
    # )
    # print(f'sample_out: {sample_output}')
    # print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

    # print_header('deactivate top_k sampling and sample only from 92% most likely words')
    # torch.random.manual_seed(3)
    # sample_output = model.generate(
    #     input_ids,
    #     do_sample=True,
    #     max_length=50,
    #     top_p=0.92,
    #     top_k=0
    # )
    # print(f'sample_out: {sample_output}')
    # print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

    # print_header('set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3')
    # torch.random.manual_seed(0)
    # sample_outputs = model.generate(
    #     input_ids,
    #     do_sample=True,
    #     max_length=50,
    #     top_k=50,
    #     top_p=0.95,
    #     num_return_sequences=3
    # )
    # print(f'sample_outputs: {sample_outputs}')
    # for i, sample_output in enumerate(sample_outputs):
    #     print(f'{i}) {tokenizer.decode(sample_output, skip_special_tokens=True)}')

    print_header('create eol')
    end_of_line = tokenizer('\n').input_ids[0]
    print(f'EOL: {end_of_line}')

    # print_header('with EOL')
    # beam_outputs = model.generate(
    #     input_ids,
    #     max_length=50,
    #     num_beams=5,
    #     no_repeat_ngram_size=2,
    #     num_return_sequences=5,
    #     early_stopping=True,
    #     eos_token_id=end_of_line
    # )
    # print(f'beam_outputs: {beam_outputs}')
    # for i, beam_output in enumerate(beam_outputs):
    #     print(f'{i}) {tokenizer.decode(beam_output, skip_special_tokens=True)}')

    # print_header('with eol, bad_words_ids')
    # beam_outputs = model.generate(
    #     input_ids,
    #     max_length=50,
    #     num_beams=5,
    #     no_repeat_ngram_size=2,
    #     num_return_sequences=5,
    #     early_stopping=True,
    #     eos_token_id=end_of_line,
    #     bad_words_ids=tokenizer(['sure', 'think', 'thundersnatch'], add_prefix_space=True)['input_ids']
    # )
    # print(f'beam_outputs: {beam_outputs}')
    # for i, beam_output in enumerate(beam_outputs):
    #     print(f'{i}) {tokenizer.decode(beam_output, skip_special_tokens=True)}')


    # print(tokenizer(['sure', ' sure', ' I am not sure'])['input_ids'])

    # print_header('with eol, bad_words_ids, force_words_ids')
    # beam_outputs = model.generate(
    #     input_ids,
    #     max_length=50,
    #     num_beams=5,
    #     no_repeat_ngram_size=2,
    #     num_return_sequences=5,
    #     early_stopping=True,
    #     eos_token_id=end_of_line,
    #     bad_words_ids=tokenizer(['sure', 'think'], add_prefix_space=True)['input_ids'],
    #     force_words_ids=[tokenizer(['cat'], add_prefix_space=True, add_special_tokens=False).input_ids],
    # )
    # print(f'beam_outputs: {beam_outputs}')
    # for i, beam_output in enumerate(beam_outputs):
    #     print(f'{i}) {tokenizer.decode(beam_output, skip_special_tokens=True)}')

    print_header('with eol, bad_words_ids, force_words_ids')
    beam_outputs = model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True,
        eos_token_id=end_of_line,
        bad_words_ids=tokenizer(['sure', 'think'], add_prefix_space=True)['input_ids'],
        force_words_ids=[
            tokenizer(['cat', 'cats', 'kitten', 'feline', 'Cat', 'Cats'], add_prefix_space=True,
                      add_special_tokens=False).input_ids,
            tokenizer(['mouse', 'mice'], add_prefix_space=True, add_special_tokens=False).input_ids,
        ],
    )
    print(f'beam_outputs: {beam_outputs}')
    for i, beam_output in enumerate(beam_outputs):
        print(f'{i}) {tokenizer.decode(beam_output, skip_special_tokens=True)}')


if __name__ == '__main__':
    run()
