
def compute_pack_ranges(threads_quantity: int, links_quantity: int) -> list:
    div = links_quantity // threads_quantity
    result = [[i * threads_quantity, i * threads_quantity + threads_quantity] for i in range(0, div)]

    if links_quantity % threads_quantity != 0:
        result.append([threads_quantity * div, links_quantity])

    return result


class LinksDS:
    def __init__(self,
                 threads_quantity: int,
                 links_creator,
                 ranges_computer=compute_pack_ranges):
        links = links_creator()
        ranges = ranges_computer(threads_quantity, len(links))
        self._link_packs = [links[r[0]:r[1]] for r in ranges]

    @property
    def link_packs(self):
        return self._link_packs

    @link_packs.setter
    def link_packs(self, value):
        raise Exception('[LinksDS] setting unsupported')
