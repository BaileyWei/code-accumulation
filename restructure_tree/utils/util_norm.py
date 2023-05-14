from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd


def fix_date(date, processing=None):
    if not processing:
        return date
    else:
        if processing == "version_date":
            convert_date = datetime.strptime(date, "%Y%m%d").strftime("%d-%b-%y")
            return convert_date
        elif processing == "text_extract":
            convert_date = datetime.strptime(date, "%B %d, %Y").strftime("%d-%b-%y")
            return convert_date
        elif processing == "adjourn_date":
            convert_date = datetime.strptime(date, "%Y-%m-%d").strftime("%d-%b-%y")
            return convert_date


def processing_normcite(cite):
    cite = cite.replace("Subarticle", "Subart.")
    cite = cite.replace("Article", "Art.")
    cite = cite.replace("Subchapter", "Subch.")
    cite = cite.replace("Chapter", "Ch.")
    cite = cite.replace("Part", "Pt.")
    cite = cite.replace("Subpart", "Subpt.")
    return cite

def get_basic_info(text):
    action = text.get('action', " ")
    section = text.get('section', " ")
    book = text.get('book', " ")
    eff_date = text.get('effective', " ")
    if eff_date != " ":
        eff_date = datetime.strptime(eff_date, "%B %d, %Y").strftime("%d-%b-%y")
    if action:
        if action == "new":
            action = "added"
        elif action == "reenacted":
            action = "repealedAndReenacted"
        elif "renumbered" in action:
            section = text.get('ref', section)
            if action == "amended and renumbered":
                action = "amendedAndRenumbered"
    return action, section, book, eff_date


class TosaBase:

    def __init__(self, **kwargs):
        file_path = kwargs.get("file_path")
        file = open(file_path)
        if not file:
            raise ValueError("xml read error")
        self.soup = BeautifulSoup(file, 'xml')
        self.preprocessing_data = kwargs.get("preprocessing_data")
        self.version_date = fix_date(self.soup.version_date.string, processing="version_date")
        self.session_id = self.soup.session_id.string
        self.year = self.session_id[:4]

    def predict(self):
        self.eff_date_map = self.get_eff_date()
        self.result_list = self.get_context_result()
        columns = ['Act Sec', 'Effect', 'Effective Date', 'Normcite', 'Subsec.']
        data = pd.DataFrame(self.result_list, columns=columns)
        data = data.drop_duplicates()
        return data.to_dict(orient='records')

    def get_nodes(self):
        nodes, parts = [], []
        text_ori = self.soup.select("billtext document > text")
        index = 0
        for text in text_ori[0].children:
            if text.name:
                node = Node(text, text.name, index)
                index += 1
                nodes.append(node)
                if text.get("class") == 'center':
                    parts.append(node)
        return nodes, parts

    def get_context_result(self):
        raise NotImplementedError

    def get_eff_date(self):
        raise NotImplementedError

    def get_normcite(self):
        raise NotImplementedError


class Node:
    def __init__(self, info, name, index):
        self.ori_info = info
        self.name = name
        self.index = index
        self.string = self.ori_info.string if self.ori_info.string else " "
        action, section, book, eff_date = get_basic_info(self.ori_info)
        self.basic_info = {"book": book,
                           "effect": action,
                           "section": section,
                           "eff_date": eff_date,
                           "normcite": " ",
                           "subsec": " ",
                           "act_sec": " ",
                           "is_acts": 0,
                           "is_omitted": 0
                           }
