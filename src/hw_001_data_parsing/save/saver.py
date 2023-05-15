import hashlib
from os import listdir, remove
from os.path import join, isfile


def rewrite_file(path: str, content: str) -> bool:
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as ex:
        print(ex)
        return False


def calculate_file_name(prefix: str, value: str) -> str:
    return prefix + str(hashlib.md5(str.encode(value)).hexdigest()) + '.htm'


class Saver:
    def __init__(self,
                 folder_path: str,
                 file_prefix: str,
                 writer=rewrite_file,
                 name_calculator=calculate_file_name) -> None:
        self._folder_path = folder_path
        self._file_prefix = file_prefix
        self._writer = writer
        self._name_calculator = name_calculator

    def save(self, key: str, content: str) -> None:
        file_name = self._name_calculator(self._file_prefix, key)
        path = f'{self._folder_path}/{file_name}'
        self._writer(path, key + '\n' + content)

    def save_dict(self, contents: dict) -> None:
        self._clear_folder()
        for k, v in contents.items():
            self.save(k, v)

    def _clear_folder(self):
        path = self._folder_path
        prefix = self._file_prefix
        files = [f for f in listdir(path) if isfile(join(path, f)) and f[:len(prefix)] == prefix]
        for f in files:
            remove(join(path, f))


if __name__ == '__main__':
    # todo del
    # path = '../../../output/feed_pages'
    # prefix = 'feed_'
    # files = [f for f in listdir(path) if isfile(join(path, f)) and f[:len(prefix)] == prefix]
    # print(files)
    #
    # for f in files:
    #     os.remove(join(path, f))

    amount = 10
    d = {'key_' + str(i): 'content ' + str(i) for i in range(0, amount)}

    saver = Saver('../../../output/feed_pages', 'prefix_')
    saver.save_dict(d)
