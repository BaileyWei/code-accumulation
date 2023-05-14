import re
import copy
from bs4 import BeautifulSoup
from utils.util_norm import fix_date
from datetime import datetime
import pandas as pd


def extract_text(node):
    """
    extract content from original plain p_node
    """
    text = ""
    if hasattr(node, 'children'):
        for c in node.children:
            if isinstance(c, str):
                text += c
            elif c.string:
                text += c.string
    return text.strip()


class TreeNode:
    def __init__(self, info, name, index, string, level=5):
        self.string = string
        self.info = info
        self.children = []
        self.name = name
        self.level = level
        self.index = index


class DC:
    def __init__(self, **kwargs):
        file_path = kwargs.get("file_path")
        file = open(file_path)
        if not file:
            raise ValueError("xml read error")
        s = ""
        for line in file.readlines():
            line = line.replace("&ldquo;", '"')
            line = line.replace("&rdquo;", '"')
            line = line.replace("&sect;", '§')
            s += line
        self.soup = BeautifulSoup(s, 'xml')
        self.preprocessing_data = kwargs.get("preprocessing_data")
        self.version_date = fix_date(self.soup.version_date.string, processing="version_date")
        self.enacted_date = self.parse_preprocessing_data()
        self.session_id = self.soup.session_id.string
        self.year = self.session_id[:4]
        self.version = self.soup.version.string.split()[-2]

    def parse_preprocessing_data(self):
        """
        parse data from database，return eff_date
        """
        enacted_date = None
        if self.preprocessing_data:
            all_actions = self.preprocessing_data.get('db_data', {}).get("actions", [])
            for s in all_actions:
                if s.get("action_flag") == 'c':
                    enacted_date = s.get("action_date")
                    break
            if enacted_date:
                try:
                    enacted_date = datetime.strptime(enacted_date, "%Y-%m-%d")
                except ValueError:
                    try:
                        enacted_date = datetime.strptime(enacted_date, "%m/%d/%Y")
                    except ValueError:
                        pass

        return enacted_date

    def predict(self):
        result_list = self.get_context_result()
        columns = ['Act Sec', 'Effect', 'Effective Date', 'Normcite', 'Subsec.']
        data = pd.DataFrame(result_list, columns=columns)
        # data = data.drop_duplicates()
        return data.to_dict(orient='records')

    def get_nodes(self):
        """
        processing p_node of xml to standard TreeNode
        """
        nodes, secs = [], []
        text_ori = self.soup.select("billtext document > text")
        for text in text_ori[0].children:
            index = len(nodes)+1
            string = extract_text(text) if extract_text(text) else " "
            if text.name and text.name == "p":
                name_info = re.search("^Sec. \d+\.|^[\"]*(\([\w\-]+\)){1,}", string)
                if name_info:
                    name = name_info[0]
                    level1 = re.search('Sec.', name)
                    level2 = re.search('[a-z]+', name)
                    level3 = re.search('\d+', name)
                    level4 = re.search('[A-Z]+', name)
                    if re.search('"', name):
                        nodes.append(TreeNode(text, "sub "+name[1:], index, string))
                    elif level1:
                        node = TreeNode(text, name.split()[-1][:-1], index, string, level=1)
                        nodes.append(node)
                        secs.append(node)
                        if len(secs) >= 2 and int(re.search("\d+", name)[0]) - 200 > int(re.search("\d+", secs[-2].name)[0]):
                            node.level = 4
                            secs.pop()
                    elif level2:
                        if level2 and level3:
                            if len(name) == 6:
                                nodes.append(TreeNode("", name[:3], index, " ", level=2))
                                nodes.append(TreeNode(text, name[3:], index+1, string, level=3))
                            else:
                                nodes.append(TreeNode(text, name, index, string))
                        else:
                            nodes.append(TreeNode(text, name, index, string, level=2))
                    elif level3:
                        nodes.append(TreeNode(text, name, index, string, level=3))
                    elif level4:
                        nodes.append(TreeNode(text, name, index, string, level=4))
                elif nodes:
                    nodes[-1].string += string
            elif text.name == "effective_clause":
                name = re.search("^Sec. \d+\.", string)[0]
                nodes.append(TreeNode(text, name.split()[-1][:-1], index, string, level=1))

        return nodes, secs

    def build_tree(self):
        """
        using the level of node and node after preprocessing
        rebuilding a structured tree of original xml
        """
        node_list, sec_list = self.get_nodes()
        n = [TreeNode('', '', 0, '', level=0)] + node_list + [TreeNode('', 'end', len(node_list)+1, '')]

        def dfs(begin, end):
            if begin == end:
                return
            level, index_list = n[begin + 1].level, []
            for node in n[begin + 1:end + 1]:
                if node.level == level:
                    index_list.append(node.index)
            index_list.append(end)
            for (left, right) in zip(index_list[:-1], index_list[1:]):
                n[begin].children.append(dfs(left, right))
            return n[begin]

        return dfs(0, len(n) - 1)

    @staticmethod
    def search_normcite(text):
        """
        extract normcite and effect from plain text
        """
        effect_info = re.search("(amended|added|repealed)", text)
        effect = effect_info[0] if effect_info else ""
        n_list = re.findall("D.C. Official Code § [0-9a-z\-\.]+", text)
        if not n_list and re.search("[sS]ection|[sS]ections|[Cc]hapter|[Tt]itle", text):
            n_list = re.findall("\d+[\-\w\.]+[\w]+", text)
            n_list = ["D.C. Code § " + n for n in n_list]
        return n_list, effect

    def get_context_result(self):
        """
        extract information from tree data
        """

        def dfs(root, name):
            if not root or root.level > 3:
                return
            # if re.search(ignore_section, root.string):
            #     return
            normcite, effect = self.search_normcite(root.string)
            if self.version == "Temporary" or self.version == "Emergency":
                if len(normcite) == 1 and effect:
                    result_list.append(
                        [name + root.name, "notedUnder", date_flag, normcite[0].replace(" Official", ""), subsec_flag])
            if self.version == "Permanent":
                if re.search("Effective date|Fiscal impact statement", root.string):
                    result_list.append([name + root.name, "excluded", date_flag, "", subsec_flag])
                    return
                elif re.search("Applicability", root.string):
                    result_list.append([name + root.name, "notedUnder", date_flag, "", subsec_flag])
                    return
                elif normcite and effect:
                    if effect == "added":
                        for norm in normcite:
                            temp = [name + root.name, effect, date_flag, norm.replace(" Official", ""), subsec_flag]
                            if temp not in result_list:
                                result_list.append(temp)
                    else:
                        result_list.append(
                            [name + root.name, effect, date_flag, normcite[0].replace(" Official", ""), subsec_flag])
                elif root.level == 1:
                    result_list.append([name + root.name, "added", date_flag, "", subsec_flag])

            for c in root.children:
                dfs(c, name + root.name)

        result_list = []
        # ignore_section = "Effective date|Fiscal impact statement|Repeals|Applicability|Transmittal"
        # excluded_section = "Reserved|budget|Budget|amount|Effective date|Fiscal impact statement"
        xml_root = self.build_tree()
        if self.version == "Temporary":
            subsec_flag = 'temp'
        else:
            subsec_flag = ''
        if self.version == "Emergency":
            date_flag = self.version_date
        else:
            if self.enacted_date:
                date_flag = self.enacted_date
            else:
                date_flag = "pending"
        dfs(xml_root, "")

        if self.version == "Permanent":
            norm_list, apply_flag = [], 0
            for i, r in enumerate(result_list):
                if r[-2]:
                    norm_list.append(r[-2])
                if r[1] == "notedUnder":
                    apply_flag = 1
                    apply_record = result_list.pop(i)

            if apply_flag:
                for norm in norm_list:
                    l = copy.deepcopy(apply_record)
                    l[-2] = norm
                    result_list.append(l)

        return result_list

if __name__ == "__main__":
    file_path = r".\sample.xml"
    # file_path = r"\\lngshadatp001\EPD\Taylor\statenet\xmlbill\DC\2019000\B\437\BILLTEXT_20210202_0_EP.xml"
    dc = DC(file_path=file_path)
    output = dc.predict()



