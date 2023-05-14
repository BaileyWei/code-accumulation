class Entity:
    def __init__(self, ori_ent, ent_typ, span, name=None, action=None, particle=None, idx=0):
        self.ent_typ = ent_typ
        self.ori_ent = ori_ent
        # for argument-section and argument-subsection e.g. name = '(A)';
        # for trigger-amendment e.g. name = 'added'
        self.name = name
        self.action = action  # for trigger-amendment
        self.particle = particle  # for argument-section and argument-subsection
        self.span = span
        self.idx = idx
        self.key = f'{self.span[0]};{self.span[1]};{self.idx}'

    def __repr__(self):
        if self.ent_typ in ['argument-section', 'argument-subsection']:
            return f'{self.particle}-{self.name}'
        elif self.ent_typ == 'trigger-amendment':
            return f'{self.name}-{self.ori_ent}'
        else:
            return self.ori_ent


class Relation:
    def __init__(self, subject_entity, relation=None, object_entity=None, ):
        self.subject_entity = subject_entity
        self.object_entity = object_entity
        self.relation = relation

    def __repr__(self):
        return f'{self.subject_entity}-{self.relation}-{self.object_entity}'


class Amendment:
    def __init__(self, path, subject_entity, trigger_amendment=None, position='in', target=None):
        self.path = path  # Entity list
        self.subject_entity = subject_entity  # Entity
        self.trigger_amendment = trigger_amendment  # Entity
        self.target = target  # Entity
        self.position = position  # in/before/after

    def __repr__(self):
        if self.target:
            return f"{'-'.join([p.name for p in self.path])}, {self.trigger_amendment.action}-{self.position}, {self.target.name}"
        elif self.trigger_amendment:
            return f"{'-'.join([p.name for p in self.path])}, {self.trigger_amendment.action}-{self.position}"
        else:
            return f"{'-'.join([p.name for p in self.path])}, substituted-{self.position}"
