import re
from utils import pinpoint_to_num_tuple, num_tuple_to_pinpoint, through


class PinPointProcessor:

    def process_entity(self, entity, entity_type):
        fine_grain = self.find_fine_grain(entity, entity_type)
        pinpoints = self.find_pinpoint_name(entity, fine_grain)
        pinpoints_list = self.pinpoint_complement(entity, pinpoints, fine_grain)
        return ', '.join(pinpoints_list)

    def find_fine_grain(self, entity, entity_type):
        fine_grain = None
        if entity_type == 'argument-subsection':
            particle_list = [
                r'(S|s)ubsection',
                r'(S|s)ubparagraph',
                r'(S|s)ubclause',
                r'(S|s)ubdivision',
                r'(P|p)aragraph',
                r'(D|d)ivision',
                r'(C|c)lause',
                r'(D|d)efinition'
            ]
            for particle in particle_list:
                if re.search(particle, entity):
                    fine_grain = re.search(particle, entity)[0]
                    break
            if not fine_grain:
                fine_grain = 'designator'

        if entity_type == 'argument-section':
            fine_grain = 'section'

        if entity_type == 'argument-content':
            fine_grain = 'text'

        if entity_type == 'trigger-amendment':
            fine_grain = 'action'

        if entity_type == 'argument-other-pos':
            fine_grain = 'other-pos'

        return fine_grain

    def find_pinpoint_name(self, entity, particle):
        designator_pattern = re.compile(r"\([^\s]+\)")
        definition_pattern = re.compile(r"\“.+?\”")

        t_new = self.pinpoint_filter(entity, particle)
        # (1)(b) and (2)(a)(I)
        if definition_pattern.search(t_new):
            pinpoint_list = definition_pattern.findall(t_new)

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

    def pinpoint_complement(self, text, pinpoint_list, particle):
        pinpoint_res = []
        for a, b in zip(pinpoint_list[:-1], pinpoint_list[1:]):
            if re.search(f'{re.escape(a)}(\s)*(through|to|-) {re.escape(b)}', text):
                a_num_tuple = pinpoint_to_num_tuple(a, particle.lower())
                b_num_tuple = pinpoint_to_num_tuple(b, particle.lower())
                throughs = through(a_num_tuple[:-1], b_num_tuple[:-1])
                for num_tuple in throughs:
                    p = num_tuple_to_pinpoint(num_tuple, particle, a_num_tuple[-1])
                    if p not in pinpoint_res:
                        pinpoint_res.append(p)
            else:
                if a not in pinpoint_res:
                    pinpoint_res.append(a)
                if b not in pinpoint_res:
                    pinpoint_res.append(b)

        return pinpoint_res

    @staticmethod
    def pinpoint_filter(text, particle):
        """
        filter some words that might influence the search of pinpoint
        """
        t_new = re.sub(f'{particle}(s)*', '', text)
        t_new = re.sub(r'and', '', t_new)
        t_new = re.sub(r'new', '', t_new)
        # t_new = re.sub(r'(?<=\))-\s(?=\()', ' - ', t_new)
        return t_new + ' '


if __name__ == "__main__":
    pinpoint_post_processor = PinPointProcessor()
    print(pinpoint_post_processor.process_entity('Subdivisions (6-a) , (6-b) , (8-a) , (8-b) , and (10-a)', 'argument-subsection'))
