import re
from collections import defaultdict
from .internal_classes import Entity, Relation, Amendment
from .utils import pinpoint_to_num_tuple, num_tuple_to_pinpoint, through, find_edges

name_clean_pattern = re.compile(r'[^.\s]+')


class PostProcessor:
    def __init__(self, ):
        self.entity_processor = EntityPostProcessor()

    def inference(self, data):
        relation_dict, entity_dict = self.processing_rel_and_ent(data)
        pinpoints = self.get_pinpoint_subjects(relation_dict, entity_dict)
        contents = self.get_content_subjects(relation_dict, entity_dict, pinpoints)
        amendment_pinpoints = self.amendment_inference(pinpoints, relation_dict, entity_dict, is_pinpoint=True)
        amendment_contents = self.amendment_inference(contents, relation_dict, entity_dict, is_pinpoint=False)
        amendments = defaultdict(list)
        amendments.update(amendment_pinpoints)
        amendments.update(amendment_contents)
        return amendments

    def processing_rel_and_ent(self, data):
        relation_dict = defaultdict(list)
        entity_dict = defaultdict()
        relation_list = data['relation_list']
        entity_list = data['entity_dict']

        for relation in relation_list:
            rels = relation['predicate']
            s, s_span, s_type = relation['subject'], relation['subj_char_span'], relation['sub_type']
            o, o_span, o_type = relation['object'], relation['obj_char_span'], relation['obj_type']
            s_list = self.entity_processor.process_entity(s, s_type, s_span, )
            o_list = self.entity_processor.process_entity(o, o_type, o_span, )
            for rel in rels:
                if rel in ['position-include', 'position-reference']:
                    if len(s_list) > 1 and len(o_list) > 1:
                        raise ValueError(r'invalid pinpoint relation')
                for s_e in s_list:
                    for o_e in o_list:
                        relation_dict[s_e.key].append(Relation(
                            subject_entity=s_e,
                            relation=rel,
                            object_entity=o_e,
                        ))
                        entity_dict[s_e.key] = s_e
                        entity_dict[o_e.key] = o_e

        for entity in entity_list:
            e, e_span, e_type = entity['text'], entity['char_span'], entity['type']
            e_list = self.entity_processor.process_entity(e, e_type, e_span, )
            for e in e_list:
                if not entity_dict.get(e.key):
                    entity_dict[e.key] = e

        return relation_dict, entity_dict

    def drop_duplicated_pinpoint(self, ):
        pass

    def amendment_inference(self, subjects, relation_dict, entity_dict, is_pinpoint=False):
        amendment_subjects = defaultdict(list)
        for k,chain in subjects.items():
            ent = entity_dict[k]
            if is_pinpoint:
                amendments = self.get_amendment(ent, relation_dict, path=chain)
                amendment_subjects[entity_dict[chain[-2].key]].extend(amendments)
            else:
                for c in chain.values():
                    amendments = self.get_amendment(ent, relation_dict, path=c)
                    amendment_subjects[entity_dict[c[-1].key]].extend(amendments)

        return amendment_subjects

    def get_pinpoint_subjects(self, relation_dict, entity_dict):
        pinpoint_chains = self.pinpoint_inference(relation_dict, entity_dict)
        pinpoints_key = list(pinpoint_chains.keys())
        for k,ent in entity_dict.items():
            if ent.ent_typ in ['argument-section', 'argument-subsection']:
                rels = relation_dict[k]
                for rel in rels:
                    if rel.relation in ['amendment-in', 'amendment-before', 'amendment-after']:
                        if k not in pinpoints_key:
                            pinpoint_chains[k] = [ent]
                            pinpoints_key.append(k)

        return pinpoint_chains

    def get_content_subjects(self, relation_dict, entity_dict, pinpoint_chains):
        content_subjects = defaultdict(list)
        list(pinpoint_chains.values())
        for k,ent in entity_dict.items():
            if ent.ent_typ == 'argument-content':
                rels = relation_dict[k]
                for rel in rels:
                    if rel.relation in ['amendment-in', 'amendment-before', 'amendment-after']:
                        content_subjects[k] = pinpoint_chains

        return content_subjects

    @staticmethod
    def pinpoint_inference(relation_dict, entity_dict):
        # 存储有向图
        pinpoint_edges = defaultdict(list)
        in_degree = defaultdict(int)
        for k, v_list in relation_dict.items():
            for v in v_list:
                if v.relation in ['position-include', 'position-reference']:
                    pinpoint_edges[k].append(v)
                    in_degree[v.object_entity.key] += 1
        #         print(pinpoint_edges)
        p_list = find_edges(entity_dict, pinpoint_edges, in_degree)
        if not p_list:
            for k,v in entity_dict.items():
                if v.ent_typ in ['argument-section', 'argument-subsection']:
                    p_list.append([(v, '', 'end')])

        pinpoint_chains = defaultdict(list)
        for p in p_list:
            pinpoint = p[-1][0]
            entity_chain = []
            for i, tuple_ in enumerate(p):
                if i > 0 and p[i - 1][1] == 'position-reference':
                    if tuple_[2].key not in entity_chain:
                        entity_chain.append(tuple_[2].key)
                elif tuple_[0].key not in entity_chain:
                    entity_chain.append(tuple_[0].key)
            pinpoint_chains[pinpoint.key] = [entity_dict[k] for k in entity_chain]

        return pinpoint_chains

    @staticmethod
    def get_amendment(subject, relation_dict, path):
        amendments = []

        if relation_dict[subject.key]:
            relation_list = relation_dict[subject.key]
            for relation in relation_list:
                if relation.object_entity.ent_typ == 'trigger-amendment':
                    amendment = Amendment(subject_entity=subject,
                                          trigger_amendment=relation.object_entity,
                                          position=relation.relation.split('-')[-1],
                                          path=path)
                    if relation_dict[relation.object_entity.key]:
                        idx = int(subject.key.split(';')[-1])
                        if idx < len(relation_dict[relation.object_entity.key]):
                            rel_1 = relation_dict[relation.object_entity.key][idx]
                            if rel_1.relation == 'amendment-target':
                                amendment.target = rel_1.object_entity
                            # fix relation error in content entity
                            elif rel_1.subject_entity.ent_typ == 'trigger-amendment' and \
                                rel_1.object_entity.ent_typ == 'argument-content':
                                amendment.target = rel_1.object_entity
                        else:
                            raise ValueError('out of idx in relation_dict')

                    amendments.append(amendment)
        else:
            return [Amendment(subject_entity=subject, path=path)]

        return amendments


class EntityPostProcessor:
    def process_entity(self, entity, entity_type, span):
        fine_grain = self.find_fine_grain(entity, entity_type)
        if entity_type == 'argument-subsection':
            pinpoints = self.find_pinpoint_name(entity, fine_grain)
            pinpoints_list = self.pinpoint_complement(entity, pinpoints, fine_grain)
            new_pinpoints_list = []
            for i, pinpoint in enumerate(pinpoints_list):
                new_pinpoints_list.append(
                    Entity(
                            ori_ent=entity,
                            ent_typ=entity_type,
                            span=span,
                            name=pinpoint,
                            particle=fine_grain.lower(),
                            idx=i,
                            )
                    )

            return new_pinpoints_list
        elif entity_type == 'argument-section':
            return [
                Entity(
                        ori_ent=entity,
                        ent_typ=entity_type,
                        span=span,
                        name=entity,
                        particle=fine_grain.lower(),
                        )
                ]

        elif entity_type == 'trigger-amendment':
            effect, action = self.find_action_and_effect(entity)
            return [
                Entity(
                    ori_ent=entity,
                    ent_typ=entity_type,
                    span=span,
                    name=effect,
                    action=action,
                )
            ]

        else:
            return [
                Entity(
                    ori_ent=entity,
                    ent_typ=entity_type,
                    span=span,
                    name=entity
                )
            ]

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
            pinpoint_list = [entity]

        return pinpoint_list

    @staticmethod
    def find_action_and_effect(entity):
        action, effect = None, None
        action_patterns = {
            'deleted': ['deleted', 'repealed', ],
            'added': ['added', ],
            'redesignated': ['redesignated', 'renumbered', 'designated', 'reenacted',],
            'substituted': ['substituted', 'revised', 'republished', ]
        }

        effect_patterns = {
            'deleted': [r'(delet|strik)', ],
            'added': [r'(insert|add)', ],
            'redesignated': [r'redesignat', r'reorder'],
            'designated': [r'designat', ],
            'renumbered': [r'renumber', ],
            'substituted': [r'(substitut|chang|amend)', r'replac', r'recreat', r'modif', 'correct'],
            'revised': [r'revis', ],
            'repealed': [r'repeal', ],
            'reenacted': [r'reenact', r'enacted in lieu thereof', ],
            'republished': [r'republish']
        }

        for act, action_list in action_patterns.items():
            for eff in action_list:
                for pattern in effect_patterns[eff]:
                    if re.search(pattern, entity):
                        effect = eff
                        action = act
                        break

        return effect, action

    @staticmethod
    def pinpoint_complement(text, pinpoint_list, particle):
        if len(pinpoint_list) <= 1:
            return pinpoint_list
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

    @staticmethod
    def find_fine_grain(entity, entity_type):
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
                r'(D|d)efinition',
                r'item'
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


class ContentUpdate:
    def __init__(self, original_content, changed_content, amendments):
        self.original_content = original_content
        self.changed_content = changed_content
        self.amendments = amendments

    def update_content(self):
        for subject, amendments in self.amendments.items():
            path = []
            for pos_ent in amendments[0].path:
                path.append(pos_ent)
                if pos_ent.key == subject.key:
                    break

            self._find_and_amend(
                amendments,
                children=self.original_content,
                path=path,
                is_root=True,)

        return self.original_content

    def _find_and_amend(self, amendments, root=None, children=None, path=None, is_root=False,):
        # find out the exact level which need to be amended
        if not root and not is_root:
            return
        if not path:
            candidate = []
            change_particle = self.get_change_particle(amendments)
            if change_particle == 'section':
                return self.changed_content
            if change_particle == 'subsection':
                candidate = self._apply_subsection_changes(
                    children,
                    amendments
                )

                if candidate:
                    if is_root:
                        return candidate
                    else:
                        root.children = candidate
            if change_particle == "content":
                pass
        else:
            n = len(children)
            for i, content in enumerate(children):
            # have designator name，use name to find out the subsection directly
                name = name_clean_pattern.search(content.local_designator)[0]
                if name == path[0].name:
                    # down to find next level
                    self._find_and_amend(
                        root=content,
                        children=content.children,
                        path=path[1:],
                    )
                    break


    def  _apply_subsection_changes(self, children, amendments):
        candidate = []
        added_amendments = []
        deleted_amendments = []
        substituted_amendments = []
        redesignated_amendments = []

        if children:
            parent_designator = children[0].full_designator[:-1]

        for amendment in amendments:
            if amendment.trigger_amendment.name == 'added':
                added_amendments.append(amendment)
            if amendment.trigger_amendment.name == 'deleted':
                deleted_amendments.append(amendment)
            if amendment.trigger_amendment.name == 'substituted':
                substituted_amendments.append(amendment)
            if amendment.trigger_amendment.name == 'redesignated':
                redesignated_amendments.append(amendment)

        if deleted_amendments:
            for am in deleted_amendments:
                for i, child in enumerate(children):
                    if child.local_designator.strip() == am.subject_entity.name:
                        child.text = 'deleted'

        if added_amendments:
            point1, point2 = 0, 0
            while point2 < len(children) and point1 < len(added_amendments):
                if (
                        children[point2].local_designator.strip()
                        == added_amendments[point1].subject_entity.name
                ):
                    for content in self.changed_content:
                        if content.local_designator == added_amendments[point1].subject_entity.name:
                            content.full_designator = parent_designator + [content.local_designator]
                            candidate.append(content)
                            break
                    point1 += 1
                else:
                    if children[point2] != 'deleted':
                        candidate.append(children[point2])
                point2 += 1
            if point1 < len(added_amendments):
                for am in added_amendments[point1:]:
                    for content in self.changed_content:
                        if content.local_designator == am.subject_entity.name:
                            content.full_designator = parent_designator + [content.local_designator]
                            candidate.append(content)
                            break
            if point2 < len(children):
                for c in children[point2:]:
                    if c.text == 'deleted':
                        continue
                    candidate.append(c)

        new_candidate = []
        if substituted_amendments:
            for can in candidate:
                substituted_flag = False
                for am in substituted_amendments:
                    if am.subject_entity.name == can.local_designator:
                        for content in self.changed_content:
                            if content.local_designator == am.subject_entity.name:
                                content.full_designator = parent_designator + [content.local_designator]
                                new_candidate.append(content)
                                substituted_flag = True
                                break
                if not substituted_flag:
                    new_candidate.append(can)

        if redesignated_amendments:
            for can in new_candidate:
                for am in redesignated_amendments:
                    if am.subject_entity.name == can.local_designator:
                        can.local_designator = am.target.name
                        can.full_designator = parent_designator + [can.local_designator]

        return new_candidate


    def get_change_particle(self, amendments):
        particles = []
        for amendment in amendments:
            if not amendment.target and amendment.subject_entity.ent_typ == 'argument-section':
                particles.append('section')
            elif not amendment.target and amendment.subject_entity.ent_typ == 'argument-subsection':
                particles.append('subsection')
            elif amendment.target and amendment.target.ent_typ == 'argument-section':
                particles.append('section')
            elif amendment.target and amendment.target.ent_typ == 'argument-subsection':
                particles.append('subsection')
            elif amendment.target and amendment.target.ent_typ == 'argument-content':
                particles.append('content')
            elif not amendment.target and amendment.subject_entity.ent_typ == 'argument-content':
                particles.append('content')

        if set(particles) == 1:
            return particles[0]



if __name__ == "__main__":
    import json
    import random

    minibatch_gpt_path = r'./pinpoint_normalization/data/labeleddata_minibatch_for_gpt.json'
    minibatch_gpt = json.load(open(minibatch_gpt_path, encoding="utf-8"))
    idx = random.sample(range(len(minibatch_gpt)), 1)[0]
    # id = 62YY-RKM1-F528-G323-00000-00_8_0
    print(minibatch_gpt[11]['text'])
    post_processor = PostProcessor()
    amendments = post_processor.inference(minibatch_gpt[11])
    for k, v in amendments.items():
        print(k)
        print(v)
