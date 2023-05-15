from typing import Optional, Tuple, Union
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def initial_parameter(net, initial_method=None, init_embed=False):
    r"""A method used to initialize the weights of PyTorch models.
    :param net: a PyTorch model
    :param str initial_method: one of the following initializations.
            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform
    :param bool init_embed: whether to initialize the embedding layer.
    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        if m.__class__.__name__ == net.__class__.__name__:
            pass  # 防止再初始化一遍自己
        elif isinstance(m, nn.Embedding):
            if not init_embed:
                print('embedding layer will not be initalized.')
                pass  # 不初始化embedding层
            else:
                embed_dim = m.weight.embedding_dim
                init.uniform_(m.weight.data, -1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim))
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            if len(m.weight.size()) > 1:
                init_method(m.weight.data)
            else:
                init.normal_(m.weight.data)  # batchnorm or layernorm
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias

    net.apply(weights_init)


def _get_encoding_type_from_tag_vocab(tag_vocab: dict) -> str:
    r"""
    给定Vocabulary自动判断是哪种类型的encoding, 支持判断bmes, bioes, bmeso, bio
    :param tag_vocab: 支持传入tag Vocabulary; 或者传入形如{0:"O", 1:"B-tag1"}，即index在前，tag在后的dict。
    :return:
    """
    tag_set = set()
    for idx, tag in tag_vocab.items():
        tag = tag[:1].lower()
        tag_set.add(tag)
    bmes_tag_set = set('bmes')
    if tag_set == bmes_tag_set:
        return 'bmes'
    bio_tag_set = set('bio')
    if tag_set == bio_tag_set:
        return 'bio'
    bmeso_tag_set = set('bmeso')
    if tag_set == bmeso_tag_set:
        return 'bmeso'
    bioes_tag_set = set('bioes')
    if tag_set == bioes_tag_set:
        return 'bioes'
    raise RuntimeError("encoding_type cannot be inferred automatically. Only support "
                       "'bio', 'bmes', 'bmeso', 'bioes' type.")


def _check_tag_vocab_and_encoding_type(tag_vocab: dict, encoding_type: str):
    r"""
    检查vocab中的tag是否与encoding_type是匹配的
    :param tag_vocab: 支持传入tag Vocabulary; 或者传入形如{0:"O", 1:"B-tag1"}，即index在前，tag在后的dict。
    :param encoding_type: bio, bmes, bioes, bmeso
    :return:
    """
    tag_set = set()
    for idx, tag in tag_vocab.items():
        tag = tag[:1].lower()
        tag_set.add(tag)

    tags = encoding_type
    for tag in tag_set:
        assert tag in tags, f"{tag} is not a valid tag in encoding type:{encoding_type}. Please check your " \
                            f"encoding_type."
        tags = tags.replace(tag, '')  # 删除该值
    if tags:  # 如果不为空，说明出现了未使用的tag
        warnings.warn(f"Tag:{tags} in encoding type:{encoding_type} is not presented in your Vocabulary. Check your "
                      "encoding_type.")


def allowed_transitions(tag_vocab: dict, encoding_type=None, include_start_end=False):
    r"""
    给定一个id到label的映射表，返回所有可以跳转的(from_tag_id, to_tag_id)列表。
    :param ~fastNLP.Vocabulary,dict tag_vocab: 支持类型为tag或tag-label。只有tag的,比如"B", "M"; 也可以是"B-NN", "M-NN",
        tag和label之间一定要用"-"隔开。如果传入dict，格式需要形如{0:"O", 1:"B-tag1"}，即index在前，tag在后。
    :param str encoding_type: 支持"bio", "bmes", "bmeso", "bioes"。默认为None，通过vocab自动推断
    :param bool include_start_end: 是否包含开始与结尾的转换。比如在bio中，b/o可以在开头，但是i不能在开头；
        为True，返回的结果中会包含(start_idx, b_idx), (start_idx, o_idx), 但是不包含(start_idx, i_idx);
        start_idx=len(id2label), end_idx=len(id2label)+1。为False, 返回的结果中不含与开始结尾相关的内容
    :return: List[Tuple(int, int)]], 内部的Tuple是可以进行跳转的(from_tag_id, to_tag_id)。
    """
    if encoding_type is None:
        encoding_type = _get_encoding_type_from_tag_vocab(tag_vocab)
    else:
        encoding_type = encoding_type.lower()
        _check_tag_vocab_and_encoding_type(tag_vocab, encoding_type)

    id_label_lst = list(tag_vocab.items())

    num_tags = len(tag_vocab)
    start_idx = num_tags
    end_idx = num_tags + 1
    allowed_trans = []
    if include_start_end:
        id_label_lst += [(start_idx, 'start'), (end_idx, 'end')]

    def split_tag_label(from_label):
        from_label = from_label.lower()
        if from_label in ['start', 'end']:
            from_tag = from_label
            from_label = ''
        else:
            from_tag = from_label[:1]
            from_label = from_label[2:]
        return from_tag, from_label

    for from_id, from_label in id_label_lst:
        from_tag, from_label = split_tag_label(from_label)
        for to_id, to_label in id_label_lst:
            to_tag, to_label = split_tag_label(to_label)
            if _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
                allowed_trans.append((from_id, to_id))
    return allowed_trans


def _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
    r"""
    :param str encoding_type: 支持"BIO", "BMES", "BEMSO", 'bioes'。
    :param str from_tag: 比如"B", "M"之类的标注tag. 还包括start, end等两种特殊tag
    :param str from_label: 比如"PER", "LOC"等label
    :param str to_tag: 比如"B", "M"之类的标注tag. 还包括start, end等两种特殊tag
    :param str to_label: 比如"PER", "LOC"等label
    :return: bool，能否跃迁
    """
    if to_tag == 'start' or from_tag == 'end':
        return False
    encoding_type = encoding_type.lower()
    if encoding_type in ('bio', 'bi'):
        r"""
        第一行是to_tag, 第一列是from_tag. y任意条件下可转，-只有在label相同时可转，n不可转
        +-------+---+---+---+-------+-----+
        |       | B | I | O | start | end |
        +-------+---+---+---+-------+-----+
        |   B   | y | - | y | n     | y   |
        +-------+---+---+---+-------+-----+
        |   I   | y | - | y | n     | y   |
        +-------+---+---+---+-------+-----+
        |   O   | y | n | y | n     | y   |
        +-------+---+---+---+-------+-----+
        | start | y | n | y | n     | n   |
        +-------+---+---+---+-------+-----+
        | end   | n | n | n | n     | n   |
        +-------+---+---+---+-------+-----+
        """
        if from_tag == 'start':
            return to_tag in ('b', 'o')
        elif from_tag in ['b', 'i']:
            return any([to_tag in ['end', 'b', 'o'], to_tag == 'i' and from_label == to_label])
        elif from_tag == 'o':
            return to_tag in ['end', 'b', 'o']
        else:
            raise ValueError("Unexpect tag {}. Expect only 'B', 'I', 'O'.".format(from_tag))

    elif encoding_type == 'bmes':
        r"""
        第一行是to_tag, 第一列是from_tag，y任意条件下可转，-只有在label相同时可转，n不可转
        +-------+---+---+---+---+-------+-----+
        |       | B | M | E | S | start | end |
        +-------+---+---+---+---+-------+-----+
        |   B   | n | - | - | n |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        |   M   | n | - | - | n |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        |   E   | y | n | n | y |   n   |  y  |
        +-------+---+---+---+---+-------+-----+
        |   S   | y | n | n | y |   n   |  y  |
        +-------+---+---+---+---+-------+-----+
        | start | y | n | n | y |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        |  end  | n | n | n | n |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        """
        if from_tag == 'start':
            return to_tag in ['b', 's']
        elif from_tag == 'b':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag == 'm':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag in ['e', 's']:
            return to_tag in ['b', 's', 'end']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'M', 'E', 'S'.".format(from_tag))

    elif encoding_type == 'bmeso':
        if from_tag == 'start':
            return to_tag in ['b', 's', 'o']
        elif from_tag == 'b':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag == 'm':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag in ['e', 's', 'o']:
            return to_tag in ['b', 's', 'end', 'o']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'M', 'E', 'S', 'O'.".format(from_tag))

    elif encoding_type in ('bioes', 'bies'):
        if from_tag == 'start':
            return to_tag in ['b', 's', 'o']
        elif from_tag == 'b':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag == 'i':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag in ['e', 's', 'o']:
            return to_tag in ['b', 's', 'end', 'o']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'I', 'E', 'S', 'O'.".format(from_tag))
    else:
        raise ValueError("Only support BIO, BMES, BMESO, BIOES encoding type, got {}.".format(encoding_type))


# implement of crf
class ConditionalRandomField(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags, include_start_end_trans=False, allowed_transitions=None,
                 initial_method=None, ignore_index=-100):
        r"""
        :param int num_tags: 标签的数量
        :param bool include_start_end_trans: 是否考虑各个tag作为开始以及结尾的分数。
        :param List[Tuple[from_tag_id(int), to_tag_id(int)]] allowed_transitions: 内部的Tuple[from_tag_id(int),
                                   to_tag_id(int)]视为允许发生的跃迁，其他没有包含的跃迁认为是禁止跃迁，可以通过
                                   allowed_transitions()函数得到；如果为None，则所有跃迁均为合法
        :param str initial_method: 初始化方法。见initial_parameter
        """
        super(ConditionalRandomField, self).__init__()

        self.include_start_end_trans = include_start_end_trans
        self.num_tags = num_tags
        self.ignore_index = ignore_index

        # the meaning of entry in this matrix is (from_tag_id, to_tag_id) score
        self.trans_m = nn.Parameter(torch.randn(num_tags, num_tags))
        if self.include_start_end_trans:
            self.start_scores = nn.Parameter(torch.randn(num_tags))
            self.end_scores = nn.Parameter(torch.randn(num_tags))

        if allowed_transitions is None:
            constrain = torch.zeros(num_tags + 2, num_tags + 2)
        else:
            constrain = torch.full((num_tags + 2, num_tags + 2), fill_value=-10000.0, dtype=torch.float)
            has_start = False
            has_end = False
            for from_tag_id, to_tag_id in allowed_transitions:
                constrain[from_tag_id, to_tag_id] = 0
                if from_tag_id == num_tags:
                    has_start = True
                if to_tag_id == num_tags + 1:
                    has_end = True
            if not has_start:
                constrain[num_tags, :].fill_(0)
            if not has_end:
                constrain[:, num_tags + 1].fill_(0)
        self._constrain = nn.Parameter(constrain, requires_grad=False)

        initial_parameter(self, initial_method)

    def _normalizer_likelihood(self, logits: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        r"""
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        :param logits:FloatTensor, max_len x batch_size x num_tags
        :param mask:BoolTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, n_tags = logits.size()
        alpha = logits[0]
        if self.include_start_end_trans:
            alpha = alpha + self.start_scores.view(1, -1)

        flip_mask = mask.eq(False)

        for i in range(1, seq_len):
            emit_score = logits[i].view(batch_size, 1, n_tags)
            trans_score = self.trans_m.view(1, n_tags, n_tags)
            tmp = alpha.view(batch_size, n_tags, 1) + emit_score + trans_score
            alpha = torch.logsumexp(tmp, 1).masked_fill(flip_mask[i].view(batch_size, 1), 0) + alpha.masked_fill(
                mask[i].eq(True).view(batch_size, 1), 0)

        if self.include_start_end_trans:
            alpha = alpha + self.end_scores.view(1, -1)

        return torch.logsumexp(alpha, 1)

    def _gold_score(self, logits, tags, mask):
        r"""
        Compute the score for the gold path.
        :param logits: FloatTensor, max_len x batch_size x num_tags
        :param tags: LongTensor, max_len x batch_size
        :param mask: ByteTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, _ = logits.size()
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)

        # trans_score [L-1, B]
        mask = mask.eq(True)
        flip_mask = mask.eq(False)
        trans_score = self.trans_m[tags[:seq_len - 1], tags[1:]].masked_fill(flip_mask[1:, :], 0)
        # emit_score [L, B]
        emit_score = logits[seq_idx.view(-1, 1), batch_idx.view(1, -1), tags].masked_fill(flip_mask, 0)
        # score [L-1, B]
        score = trans_score + emit_score[:seq_len - 1, :]
        score = score.sum(0) + emit_score[-1].masked_fill(flip_mask[-1], 0)
        if self.include_start_end_trans:
            st_scores = self.start_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[0]]
            last_idx = mask.long().sum(0) - 1
            ed_scores = self.end_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[last_idx, batch_idx]]
            score = score + st_scores + ed_scores
        # return [B,]
        return score

    def _forward(self, feats, tags, mask):
        r"""
        用于计算CRF的前向loss，返回值为一个batch_size的FloatTensor，可能需要mean()求得loss。
        :param torch.FloatTensor feats: batch_size x max_len x num_tags，特征矩阵。
        :param torch.LongTensor tags: batch_size x max_len，标签矩阵。
        :param torch.ByteTensor mask: batch_size x max_len，为0的位置认为是padding。
        :return: torch.FloatTensor, (batch_size,)
        """
        feats = feats.transpose(0, 1)
        tags = tags.transpose(0, 1).long()
        mask = mask.transpose(0, 1).float()
        all_path_score = self._normalizer_likelihood(feats, mask)
        gold_path_score = self._gold_score(feats, tags, mask)

        return all_path_score - gold_path_score

    def forward(self, feats, tags=None, mask=None):
        r"""
        用于计算CRF的前向loss，返回值为一个batch_size的FloatTensor，可能需要mean()求得loss。
        :param torch.FloatTensor feats: batch_size x max_len x num_tags，特征矩阵。
        :param torch.LongTensor tags: batch_size x max_len，标签矩阵。
        :param torch.ByteTensor mask: batch_size x max_len，为0的位置认为是padding。
        :return: torch.FloatTensor, (batch_size,)
        """
        if tags is None:
            return self.viterbi_decode(feats, mask)
        else:
            if mask is None:
                mask = tags.eq(self.ignore_index)
            return self._forward(feats, tags, mask)

    def viterbi_decode(self, logits, mask, unpad=False):
        r"""
        给定一个特征矩阵以及转移分数矩阵，计算出最佳的路径以及对应的分数
        :param torch.FloatTensor logits: batch_size x max_len x num_tags，特征矩阵。
        :param torch.ByteTensor mask: batch_size x max_len, 为0的位置认为是pad；如果为None，则认为没有padding。
        :param bool unpad: 是否将结果删去padding。False, 返回的是batch_size x max_len的tensor; True，返回的是
            List[List[int]], 内部的List[int]为每个sequence的label，已经除去pad部分，即每个List[int]的长度是这
            个sample的有效长度。
        :return: 返回 (paths, scores)。
                    paths: 是解码后的路径, 其值参照unpad参数.
                    scores: torch.FloatTensor, size为(batch_size,), 对应每个最优路径的分数。
        """
        batch_size, max_len, n_tags = logits.size()
        seq_len = mask.long().sum(1)
        logits = logits.transpose(0, 1).data  # L, B, H
        mask = mask.transpose(0, 1).data.eq(True)  # L, B
        flip_mask = mask.eq(False)

        # dp
        vpath = logits.new_zeros((max_len, batch_size, n_tags), dtype=torch.long)
        vscore = logits[0]  # bsz x n_tags
        transitions = self._constrain.data.clone()
        transitions[:n_tags, :n_tags] += self.trans_m.data
        if self.include_start_end_trans:
            transitions[n_tags, :n_tags] += self.start_scores.data
            transitions[:n_tags, n_tags + 1] += self.end_scores.data

        vscore += transitions[n_tags, :n_tags]

        trans_score = transitions[:n_tags, :n_tags].view(1, n_tags, n_tags).data
        end_trans_score = transitions[:n_tags, n_tags + 1].view(1, 1, n_tags).repeat(batch_size, 1, 1)  # bsz, 1, n_tags

        # 针对长度为1的句子
        vscore += transitions[:n_tags, n_tags + 1].view(1, n_tags).repeat(batch_size, 1) \
            .masked_fill(seq_len.ne(1).view(-1, 1), 0)
        for i in range(1, max_len):
            prev_score = vscore.view(batch_size, n_tags, 1)
            cur_score = logits[i].view(batch_size, 1, n_tags) + trans_score
            score = prev_score + cur_score.masked_fill(flip_mask[i].view(batch_size, 1, 1), 0)  # bsz x n_tag x n_tag
            # 需要考虑当前位置是该序列的最后一个
            score += end_trans_score.masked_fill(seq_len.ne(i + 1).view(-1, 1, 1), 0)

            best_score, best_dst = score.max(1)
            vpath[i] = best_dst
            # 由于最终是通过last_tags回溯，需要保持每个位置的vscore情况
            vscore = best_score.masked_fill(flip_mask[i].view(batch_size, 1), 0) + vscore.masked_fill(
                mask[i].view(batch_size, 1), 0)
        # backtrace
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(max_len, dtype=torch.long, device=logits.device)
        lens = (seq_len - 1)
        # idxes [L, B], batched idx from seq_len-1 to 0
        idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % max_len

        ans = logits.new_empty((max_len, batch_size), dtype=torch.long)
        ans_score, last_tags = vscore.max(1)
        ans[idxes[0], batch_idx] = last_tags
        for i in range(max_len - 1):
            last_tags = vpath[idxes[i], batch_idx, last_tags]
            ans[idxes[i + 1], batch_idx] = last_tags
        ans = ans.transpose(0, 1)
        if unpad:
            paths = []
            for idx, max_len in enumerate(lens):
                paths.append(ans[idx, :max_len + 1].tolist())
        else:
            paths = ans
        return paths, ans_score

