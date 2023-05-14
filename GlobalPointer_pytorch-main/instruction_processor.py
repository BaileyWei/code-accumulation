import re
from copy import deepcopy


class TextProcessor:
    def __init__(self, train=False, typ='span'):
        self.train = train
        self.typ = typ

    def get_processed_text(self, text, cite_mention=None, jx_name=None):
        if self.train:
            res = self.find_label(text, cite_mention=cite_mention, jx_name=jx_name)

            return {
                'text': res['context'],
                'label': res['span_posLabel']
            }

        else:
            processed_text = self.add_space_before_punctuation(text)

            return {
                'text': processed_text,
                'label': None
            }

    def find_label(self, text, cite_mention=None, jx_name=None):
        labels = []
        designator_lists = self.find_designator(self._process_quote(text)[0], jx_name=jx_name)
        designator_lists = self.designator_postprocessing(designator_lists, jx_name=jx_name)
        if cite_mention:
            section_lists = self.find_section(text, cite_mention)
        else:
            section_lists = []
        labels.extend(designator_lists)
        labels.extend(section_lists)
        res = self.idx2span(text, labels)

        return res

    def idx2span(self, text, idx_lists):
        ori_text = self.add_space_before_punctuation(text)
        dict_ = {'context': ori_text,
                 'span_posLabel': {}}

        for idx in idx_lists:
            particle = idx[2]
            text1 = self.add_space_before_punctuation(text[:idx[1][0]])

            word_b_idx = len(text1.split())
            word_e_idx = word_b_idx + len(self.add_space_before_punctuation(idx[0]).split()) - 1
            if particle == 'designator':
                if ori_text[word_b_idx - 1].lower() == 'section':
                    continue
            if re.search(r'\)\)', ' '.join(ori_text.split()[word_b_idx:word_e_idx + 1])):
                continue
            if re.search(r'\)\”', ' '.join(ori_text.split()[word_b_idx:word_e_idx + 1])):
                continue

            if ori_text.split()[word_b_idx:word_e_idx + 1] == [','] and particle == 'section':
                word_b_idx = word_b_idx - 1
                word_e_idx = word_e_idx - 1

            key = f"{word_b_idx};{word_e_idx}"
            if self.typ == 'idx':
                str_b_idx = len(' '.join(ori_text.split()[:word_b_idx])) + 1
                str_e_idx = len(' '.join(ori_text.split()[:word_e_idx + 1]))
                key = f"{str_b_idx};{str_e_idx}"
            dict_['span_posLabel'][key] = particle

        return dict_

    def find_designator(self, text, idx=0, span_list=None, depth=0, jx_name=None):
        if not span_list:
            new_span_list = []
        else:
            new_span_list = deepcopy(span_list)

        level = r'subsection|subsections|subparagraph|subparagraphs|clause|clauses|paragraph|paragraphs|subdivision|subdivisions|division|divisions'
        # (a), (d), (e) and (j)
        pattern1 = re.compile(
            r'((%s)(\s))*(\([^\s\:\,]+\)(\,)* ){1,}((and |- |to |through )(\([^\s\:\)\,]+\))+)*' % (level))
        #     pattern1 = re.compile(r'((%s)(\s))*(\([^\s\:]+\)(\,)* ){1,}((and |- |to |through )(\([^\s\:\)]+\))+)*' % (level))

        # 2, 3, 4, and 5-c
        pattern2 = re.compile(
            r'(%s) ([^\s\:]{1,4}(\,)* ){1,}((and |- |to |through )[^\s\:]{1,4})*(\s)*(?=(of|as|is|are|to))*' % (level))
        # other single pattern
        pattern3 = re.compile(r'the section heading')
        pattern4 = re.compile(r'(a|the) opening paragraph')
        pattern5 = re.compile(r'(a|the) closing paragraph')
        pattern6 = re.compile(r'the first blocked paragraph')
        pattern7 = re.compile(r'(%s) ([^\s\:]{1,4})(?=( and |,))' % (level))
        pattern8 = re.compile(r'((%s)(\s))*\([^\s\:\,]+\) through \([^\s\:\,]+\)' % (level))
        #     pattern8 = re.compile(r'((%s)(\s))*\([^\s\:]+\) through \([^\s\:]+\)' % (level))

        pattern9 = re.compile(r'(%s) [^\s\:]{1,4} through [^\s\:]{1,4}' % (level))
        pattern10 = re.compile(r'sub-§[^\s\:]{1,4}(?=\,)')

        flag = False
        for pattern in [pattern3, pattern4, pattern5, pattern6, pattern8, pattern9, pattern1, pattern2, pattern7, ]:
            if pattern.search(text.lower()):
                flag = True
                break
        if not flag:
            # (\([^\#\)]+\))+
            # \([^\#\)]+\)*
            if re.search(r'(\([^\#\)\,\s]+\))+', text.lower()):
                start_index = re.search(r'(\([^\#\)\,\s]+\))+', text.lower()).span(0)[0]
                end_index = re.search(r'(\([^\#\)\,\s]+\))+', text.lower()).span(0)[1]
                new_span_list.append((text[start_index:end_index], (start_index, end_index),))
                text = text[:start_index] + '#' * len((text[start_index:end_index + 1])) + text[end_index + 1:]
                new_span_list = self.find_designator(text,
                                                     span_list=new_span_list,
                                                     depth=depth + 1,
                                                     idx=idx,
                                                     jx_name=jx_name)
            if jx_name == 'ME' and pattern10.search(text.lower()):
                start_index = pattern10.search(text.lower()).span(0)[0]
                end_index = pattern10.search(text.lower()).span(0)[1]
                new_span_list.append((text[start_index:end_index], (start_index, end_index),))
                text = text[:start_index] + '#' * len((text[start_index:end_index + 1])) + text[end_index + 1:]
                new_span_list = self.find_designator(text,
                                                     span_list=new_span_list,
                                                     depth=depth + 1,
                                                     idx=idx,
                                                     jx_name=jx_name)
            return new_span_list

        pattern = [pattern3, pattern4, pattern5, pattern6, pattern8, pattern9, pattern1, pattern2, pattern7, ][idx]
        #         print(text)
        if pattern.search(text.lower()):
            start_index = pattern.search(text.lower()).span(0)[0]
            end_index = pattern.search(text.lower()).span(0)[1]
#             print(pattern.search(text.lower())[0])
#         if re.search(level,  pattern.search(text.lower())[0]) and len(pattern.search(text.lower())[0].split()) == 1:
#             continue
            if pattern.search(text.lower())[0].endswith(' '):
                start_index = pattern.search(text.lower()).span(0)[0]
                end_index = pattern.search(text.lower()).span(0)[1]
                end_index = end_index - 1
            if pattern.search(text.lower())[0].endswith(' and '):
                start_index = pattern.search(text.lower()).span(0)[0]
                end_index = pattern.search(text.lower()).span(0)[1]
                end_index = end_index - 4
            if re.search(r'.*(?= (of|as|is|are|to) )', pattern.search(text.lower())[0]):
                start_index = pattern.search(text.lower()).span(0)[0]
                end_index = pattern.search(text.lower()).span(0)[1]
                end_index = start_index + len(
                    re.search(r'.*(?= (of|as|is|are|to) )', pattern.search(text.lower())[0])[0])
            #             print(text[start_index:end_index])
            if pattern.search(text.lower())[0].endswith(' to read as '):
                print(pattern.search(text.lower())[0])
                start_index = pattern.search(text.lower()).span(0)[0]
                end_index = pattern.search(text.lower()).span(0)[1]
                end_index = end_index - 12
            if re.search(r'^(\s)*introductory portion', text[end_index:]):
                start_index = pattern.search(text.lower()).span(0)[0]
                end_index = pattern.search(text.lower()).span(0)[1]
                end_index = end_index + len(re.search(r'^(\s)*introductory portion', text[end_index:])[0])

            if re.search(r'([A-Za-z0-9]{1,3}\. )+', text[0:start_index]):
                #             print(re.search(r'([A-Za-z0-9]{1,2}. )+', text[0:start_index]).span(0), start_index)
                if re.search(r'([A-Za-z0-9]{1,3}\. )+', text[0:start_index]).span(0)[1] == start_index:
                    #                 print()
                    start_index = re.search(r'([A-Za-z0-9]{1,3}\. )+', text[0:start_index]).span(0)[0]
            if re.search(r'^(\s)*([A-Za-z0-9]{1,3}\. )+', text[end_index:]):
                end_index = end_index + re.search(r'^(\s)*([A-Za-z0-9]{1,3}\. )+', text[end_index:]).span(0)[1] - 1

            new_span_list.append((text[start_index:end_index], (start_index, end_index),))
            #             print(text[start_index:end_index+1])
            #             print(text)
            text = text[:start_index] + '#' * len((text[start_index:end_index + 1])) + text[end_index + 1:]

        if pattern.search(text.lower()):
            idx = idx
        else:
            idx = idx + 1

        new_span_list = self.find_designator(text,
                                             span_list=new_span_list,
                                             depth=depth + 1,
                                             idx=idx,
                                             jx_name=jx_name)

        return new_span_list

    @staticmethod
    def designator_postprocessing(span_list, jx_name=None):
        level = r'(subsection|subparagraph|clause|paragraph|subdivision|division)'
        new_span_list = []
        if not span_list:
            return []
        for span in span_list:
            begin_idx = span[1][0]
            end_idx = span[1][1]
            text = span[0]
            #         print(text)
            if re.search(level, span[0].lower()):
                partical = re.search(level, span[0].lower())[0]
            else:
                partical = 'designator'

            if text.endswith(','):
                text = text[:-1]

            if jx_name == 'IA':
                if re.search(r' Code [0-9]{4}', span[0]):
                    end_idx = re.search(r' Code [0-9]{4}', span[0]).span(0)[0] - 1

            text = text[:end_idx]

            new_span_list.append((text, (begin_idx, len(text) + begin_idx), partical))
        return new_span_list

    @staticmethod
    def add_space_before_punctuation(text):
        text1 = re.sub(r'(?<=[^\s\)])\((?=[^\s])', r' (', text)
        text2 = re.sub(r'(?<=[^\s])\,', r' ,', text1)
        text3 = re.sub(r'\§(?=([0-9A-Za-z]+))', r'§ ', text2)
        if text3.endswith(':'):
            text3 = text3[:-1] + ' :'
        if text3.endswith('.'):
            text3 = text3[:-1] + ' .'
        text4 = re.sub(r'(?<=[^\s])\;', r' ;', text3)
        return text4

    @staticmethod
    def _process_quote(description):
        notes = description
        quote = re.search(r"\“.*\”", notes)
        if quote:
            _notes = ""
            quote_text = list(set(re.findall(r"\“.*?\”", notes)))
            quote_text_map = dict(
                zip(quote_text, [
                    '#'*len(text) for text in quote_text])
            )
            quote_text_map2 = dict(
                zip(quote_text_map.values(), quote_text_map.keys())
            )
            for k, v in quote_text_map.items():
                notes = notes.replace(k, v)
            return notes, quote_text_map, quote_text_map2
        return notes, {}, {}

    @staticmethod
    def find_section(text, span_text):
        span_list = span_text.split(', ')
        index_list = []
        for span in span_list:
            span = span.replace(' ', '')
            if re.search(re.escape(span), text):
                start_index = re.search(re.escape(span), text).span(0)[0]
                end_index = re.search(re.escape(span), text).span(0)[1]
                index_list.append((span, (start_index, end_index), 'section'))
        return index_list


if __name__ == '__main__':
    from NER.inference import inference
    processor = TextProcessor(train=True, typ='idx')
    text = '24-A MRSA §222, sub-§7-A, ¶D, as enacted by PL 2013, c. 238, Pt. A, §15 and affected by §34, is amended by amending subparagraph (3) to read:'
    cite = '222'
    label = processor.get_processed_text(text, cite_mention=cite, jx_name='ME')
    print(label)
