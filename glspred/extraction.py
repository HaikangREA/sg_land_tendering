def extract_num(string: str, type: str = 'all', decimal: bool = False, ignore_sep: str = None, keep: str = None):
    # 'type' means all numbers or just num between whitespaces by specifying type='between_spaces'
    # 'ignore_sep' can be 'any' to ignore all sep, or specify a sep like ',', then func won't treat ',' as a separator
    # 'keep' allows the func to keep all matched numbers or selected ones
    import re
    import itertools

    # if the input is already int or float, return itself: input=1234 -> output=1234
    if isinstance(string, int) or isinstance(string, float):
        num = string
        return num

    else:
        string = str(string)
        # # remove all spaces from string
        # string = ''.join(string.split(' '))
        try:
            # if the string can directly be converted to number, do so (e.g. input='1234' -> output=1234.0)
            num = float(string)
            return num

        except:
            pattern = r"\d+"  # find all numbers, any digits (ignore decimal number: input='$12.3' -> output=['12','3']
            if decimal:
                pattern = r"\d*\.?\d+"  # also match decimal numbers: input='$12.3' -> output='12.3'
            if type == 'between_spaces':
                pattern = r"\b" + pattern + r"\b"
                # match numbers in between white spaces
                # input='is $10.5 per box' -> output=None; input='is 10.5 dollars per box' -> output='10.5'
            num_list = re.findall(pattern, string)

            if ignore_sep:
                if ignore_sep == 'any':  # ignore any separator between numbers
                    # input='123a456,789.654' -> output='123456789654'
                    if len(num_list) >= 1:
                        num = "".join(num_list)
                        return float(num)
                    else:
                        return np.nan
                else:
                    # ignore specified separator
                    # input='$1,234,567.05' -> output ignore ',' & decimal is T='1234567.05'
                    # output without ignoring & decimal is T=['1','234','567.05']
                    string = string.replace(ignore_sep, "")
                    num_list = re.findall(pattern, string)
            num_list = [float(num) for num in num_list]  # convert all matched str item to float, stored in list

            if keep:  # to specify certain numbers to keep by index, e.g. num_list=[5, 6, 7], keep=1 -> output=[5]
                strip = [i.split(",") for i in keep.split("-")]
                # for now only support ",", for "-" will amend this later
                keep_idx = list(set([int(i) for i in list(itertools.chain.from_iterable(strip))]))
                if len(num_list) > len(keep_idx):  # if not keeping all items, raise a msg to notify
                    print(f"{len(num_list)} numbers detected")
                num_list = [num_list[i - 1] for i in keep_idx if 0 <= i - 1 < len(num_list)]

                if len(num_list) > 0:
                    return num_list[0] if len(num_list) == 1 else num_list
                else:
                    return np.nan

            if len(num_list) == 1:
                return num_list[0]  # if the result num_list has only 1 value, output the value as float
            elif len(num_list) > 1:
                return num_list  # otherwise output the whole num_list
            else:
                return np.nan