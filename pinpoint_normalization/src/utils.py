import re
import numpy as np

definition_pattern = re.compile(r"\([^\s]+\)")
designator_pattern = re.compile(r"\“.+?\”")  # find full designator
inbracket_compile = re.compile("(?<=\\().+?(?=\\))")
SYMBOL_VALUES = {
    'I': 1,
    'V': 5,
    'X': 10,
    'L': 50,
    'C': 100,
    'D': 500,
    'M': 1000,
}


def find_pinpoint_name(text, particle):
    t_new = pinpoint_filter(text, particle)

    # (1)(b) and (2)(a)(I)
    if definition_pattern.search(t_new):
        pinpoint_list = designator_pattern.findall(t_new)

    # the definitions of “Letter of eligibility” and “Mixed use development”
    elif designator_pattern.search(t_new):
        pinpoint_list = designator_pattern.findall(t_new)

    # paragraphs o , z , bj , and bl
    elif re.search(r' [A-Za-z0-9]{1,3} ', t_new):
        pinpoints = re.findall(r' [A-Za-z0-9]{1,3} ', t_new)
        pinpoint_list = [pinpoint.strip() for pinpoint in pinpoints]

    else:
        pinpoint_list = []

    return pinpoint_list


def pinpoint_filter(text, particle):
    t_new = re.sub(f'{particle}(s)*', '', text)
    t_new = re.sub(r'and', '', t_new)
    return t_new + ' '


def pinpoint_complement(text, pinpoint_list, particle):
    pinpoint_res = []

    for a, b in zip(pinpoint_list[:-1], pinpoint_list[1:]):
        if re.search(f'{a} (through|to|-) {b}', text):
            a_num_tuple = pinpoint_to_num_tuple(a, particle.lower())
            b_num_tuple = pinpoint_to_num_tuple(b, particle.lower())
            through = through(a_num_tuple[:-1], b_num_tuple[:-1])
            for num_tuple in through:
                p = num_tuple_to_pinpoint(num_tuple, particle, a_num_tuple[-1])
                pinpoint_res.append(p)
        else:
            if a not in pinpoint_res:
                pinpoint_res.append(a)
            if b not in pinpoint_res:
                pinpoint_res.append(b)

    return pinpoint_res


def num_tuple_to_pinpoint(num_tuple_list, particle, reference):
    elements_norm = inbracket_compile.findall(reference[0])
    elements_ori = inbracket_compile.findall(reference[1])
    full_desig = []
    i = 0
    while i < len(num_tuple_list):
        local_desig = num_tuple_list[i][1]

        # e.g convert a,1 to a-1
        if elements_ori and i < len(elements_ori) and re.search(r"\-", elements_ori[i]):
            local_desig += '-' + num_tuple_list[i + 1][1]
            i += 1

        # e.g. convert a to aa
        if elements_ori and i < len(elements_ori) and len(elements_ori[i]) > 1 and len(set(elements_ori[i])) == 1:
            local_desig = local_desig * len(elements_ori[i])

        # e.g convert a,b to ab
        while i+1 < len(num_tuple_list) and \
                num_tuple_list[i][-1] == num_tuple_list[i + 1][-1] == 'char' and \
                (num_tuple_list[i][1].isupper() == num_tuple_list[i + 1][1].isupper()):
            local_desig += num_tuple_list[i + 1][1]
            i += 1

        full_desig.append(local_desig)
        i += 1

    if elements_ori:
        return f"({')('.join(full_desig)})"
    else:
        return "".join(full_desig)


def through(start, end):
    """
    input: e.g. start, end = ((2, '2', 'int'),
                              (1, 'A', 'char'),
                              (1, 'i', 'roman'),),

                             ((2, '2', 'int'),
                              (2, 'A', 'char'),
                              (3, 'iii', 'roman'),)

    output: e.g. num_tuple_list = [((2, '2', 'int'),
                                    (1, 'A', 'char'),
                                    (1, 'i', 'roman')),

                                   ((2, '2', 'int'),
                                    (1, 'A', 'char'),
                                    (2, 'ii', 'roman')),

                                   ((2, '2', 'int'),
                                    (1, 'A', 'char'),
                                    (3, 'iii', 'roman'))]
    """

    start_tuple_list = ((0,),) + start
    end_tuple_list = ((0,),) + end
    num_tuple_list = []

    def permutations(tuple_, index, res, n, is_equal_start=True, is_equal_end=True):
        """
        get pinpoint arrangement using dfs
        """
        if len(tuple_) == n:
            if tuple_ not in res:
                res.append(tuple_[1:])
            return

        pre_s = start_tuple_list[index - 1][0]
        pre_e = end_tuple_list[index - 1][0]
        cur_s = start_tuple_list[index][0]
        cur_e = end_tuple_list[index][0]
        pre_state = tuple_[index - 1][0]

        is_equal_start = is_equal_start & (pre_state == pre_s)
        is_equal_end = is_equal_end & (pre_state == pre_e)

        start = cur_s if is_equal_start else 1
        end = cur_e if is_equal_end else get_upper_limit(start_tuple_list[index][-1])

        num_list = np.arange(start, end + 1)
        for cur_state in num_list:
            t = num2tuple(cur_state, start_tuple_list[index][1], start_tuple_list[index][2])
            permutations(tuple_[:index] + (t,), index + 1, res, n, is_equal_start=is_equal_start,
                         is_equal_end=is_equal_end)

    permutations(((0,),), 1, num_tuple_list, len(end_tuple_list))

    return num_tuple_list


def get_upper_limit(typ):
    return 10 if typ != 'char' else 26


def num2tuple(num, symbol, typ):
    """
    transform num to designator according to the given symbol and typ (char/int/roman)
    """
    if typ == 'char':
        if symbol.upper() == symbol:
            tuple_ = (num, chr(num + 64), typ)
        else:
            tuple_ = (num, chr(num + 96), typ)
    if typ == 'int':
        tuple_ = (num, str(num), typ)
    if typ == 'roman':
        if symbol.upper() == symbol:
            tuple_ = (num, int2roman(num), typ)
        else:
            tuple_ = (num, int2roman(num).lower(), typ)
    return tuple_


def pinpoint_to_num_tuple(desig, particle, ):
    """
    input: e.g. (1)(A)(i), (1-A)(i)
    return: e.g. ((1, '1', 'int'),
                  (1, 'A', 'char'),
                  (1, 'i', 'roman'),
                  ('(1)(A)(i)', '(1)(A)(i)')),

                 ((1, '1', 'int'),
                  (1, 'A', 'char'),
                  (1, 'i', 'roman'),
                  ('(1)#(A)(i)', '(1-A)(i)'))

    and when 'a'-'z' are used, the new designator will be 'aa', 'bb', ...,
    """
    if not desig:
        return (0, desig)
    ori_desig = desig
    desig = re.sub(r"\(\)", "( )", desig)
    desig = re.sub(r"\$", "", desig)  # (c)($25)

    if not inbracket_compile.search(desig):
        desig = f'({desig})'
        # a-2 are not invalid
    #         raise ValueError(
    #             f"designator:{desig} format error, should be with bracket and not a empty brancket")

    desig = re.sub(r"\(\)", "( )", desig)
    if re.search(r"([^\)])\-([^\(])", desig):
        # remove "-" in designator (1)(c-1) -> (1)(c)(1)
        desig = re.sub(r"([^\)])\-([^\(])", r"\1)#(\2", desig)
    alld = inbracket_compile.findall(desig)
    alld_upper = [d.upper() for d in alld]
    n = len(alld)
    if n == 1 and alld[0] == " ":
        return ((0, " ", None), (desig, ori_desig))
    elif n == 1 and re.search(r"^PARA(\d+)$", alld[0]):
        # for para1, para2, ...
        seq = re.findall(r"^PARA(\d+)$", alld[0])[0]
        return ((int(seq), seq, 'int'), (desig, ori_desig))
    elif n == 1 and re.search(r"^DEF\=", alld[0]):
        # for def=Dualflush effective flush volume, ...
        dfn = re.findall(r"^DEF\=(.*)", alld[0])[0]
        dfn = re.findall(r"[A-Z0-9]", dfn)
        res = ()
        if dfn:
            for d in dfn:
                res += ((ord(d) - 64, d, 'char'),)
            res += ((desig, ori_desig),)
            return res
        else:
            raise ValueError(f"no word or number in {alld[0]}")
    elif n == 1 and alld[0].isdigit():
        return ((int(alld_upper[0]), alld[0], 'int'), (desig, ori_desig))
    elif n == 1 and len(alld[0]) == 1 and alld[0] != "I":
        return ((ord(alld_upper[0]) - 64, alld[0], 'char'), (desig, ori_desig))
    elif n == 1 and is_roman_number(alld_upper[0]):
        return ((roman2int(alld_upper[0]), alld[0], 'roman'), (desig, ori_desig))
    elif n == 1 and len(set(alld[0])) == 1:
        return ((ord(alld_upper[0][0]) - 64, alld[0], 'char'), (desig, ori_desig))
    elif n == 1 and len(alld[0]) > 1:
        res = ()
        for i, s in enumerate(alld_upper[0]):
            res += ((ord(s) - 64, alld[0][i], 'char'),)
        res += ((desig, ori_desig),)
        return res
    res = ()
    for i in range(n):
        if not res:
            if alld[i] == " ":
                res += ((0, alld[i], None),)
            elif alld[i].isdigit():
                res += ((int(alld_upper[i]), alld[i], 'int'),)
            elif re.search(r"^PARA(\d+)$", alld[i]):
                seq = re.findall(r"^PARA(\d+)$", alld[i])[0]
                res += ((int(seq), seq, 'int'),)
            elif re.search(r"^DEF\=", alld[i]):
                dfn = re.findall(r"^DEF\=(.*)", alld[i])[0]
                dfn = re.findall(r"[A-Z0-9]", dfn)
                if dfn:
                    for d in dfn:
                        res += ((ord(d) - 64, d, 'char'),)
            elif len(alld[i]) > 1 and is_roman_number(alld_upper[i]):
                res += ((roman2int(alld_upper[i]), alld[i], 'roman'),)
            elif len(alld[i]) == 1:
                res += ((ord(alld_upper[i]) - 64, alld[i], 'char'),)
            elif len(set(alld[i])) == 1:
                res += ((ord(alld_upper[i][0]) - 64, alld[i], 'char'),)
            elif len(alld[i]) > 1:
                for j, s in enumerate(alld_upper[i]):
                    res += ((ord(s) - 64, alld[i][j], 'char'),)
            else:
                raise ValueError(f"designator {alld[i]} is not valid")
        else:
            if alld[i].isdigit():
                res += ((int(alld_upper[i]), alld[i], 'int'),)
            elif re.search(r"^PARA(\d+)$", alld[i]):
                seq = re.findall(r"^PARA(\d+)$", alld[i])[0]
                res += ((int(seq), seq, 'int'),)
            elif re.search(r"^DEF\=", alld[i]):
                dfn = re.findall(r"^DEF\=(.*)", alld[i])[0]
                dfn = re.findall(r"[A-Z0-9]", dfn)
                if dfn:
                    for d in dfn:
                        res += ((ord(d) - 64, d, 'char'),)
            elif len(alld[i]) > 1 and is_roman_number(alld_upper[i]):
                res += ((roman2int(alld_upper[i]), alld[i], 'roman'),)
            elif len(alld[i]) == 1 and alld[i - 1].isdigit():
                res += ((ord(alld_upper[i]) - 64, alld[i], 'char'),)
            elif len(alld[i]) == 1 and is_roman_number(alld_upper[i]):
                res += ((roman2int(alld_upper[i]), alld[i], 'roman'),)
            elif len(alld[i]) == 1:
                res += ((ord(alld_upper[i]) - 64, alld[i], 'char'),)
            elif len(set(alld[i])) == 1:
                res += ((ord(alld_upper[i][0]) - 64, alld[i], 'char'),)
            elif len(alld[i]) > 1:
                for j, s in enumerate(alld_upper[i]):
                    res += ((ord(s) - 64, alld[i][j], 'char'),)
            else:
                raise ValueError(
                    f"designator {alld[i]} is not valid, full is {desig}")
    res += ((desig, ori_desig),)
    return res


def is_roman_number(number: str) -> bool:
    """
    check if is a roman number
    """
    numerals = 'IVXLCDM'
    numdict = dict(zip(numerals, range(len(numerals))))
    m = ''
    repeat = 0
    ridge = 6
    for n in number:
        if n not in numerals:
            return False
        if m == n:
            repeat += 1
            if repeat > 2:
                return False
            elif numdict[n] % 2 == 1:
                return False
        else:
            repeat = 0
        if m:
            near = numdict[n] - numdict[m]
            if near > 2:
                return False
            elif near > 0:
                if numdict[n] > ridge:
                    return False
                else:
                    ridge = numdict[n]
        m = n
    return True


def roman2int(s: str) -> int:
    """
    convert roman number to number
    """
    ans = 0
    n = len(s)
    for i, ch in enumerate(s):
        value = SYMBOL_VALUES[ch]
        if i < n - 1 and value < SYMBOL_VALUES[s[i + 1]]:
            ans -= value
        else:
            ans += value
    return ans


def int2roman(num: int) -> str:
    """
    convert number to roman number
    """
    Roman = ""
    storeIntRoman = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]]
    for i in range(len(storeIntRoman)):
        while num >= storeIntRoman[i][0]:
            Roman += storeIntRoman[i][1]
            num -= storeIntRoman[i][0]
    return Roman

