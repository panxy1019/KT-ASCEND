"""Synchronous CPU PLE hash canonical using the checkpoint's original buffers.

Matches Qwen4ExpNGramEmbedding at upstream4ccff141. Caller owns immutable
request history; no shared mutable request cache. No speculative/asynchronous path.
"""
class NGramHasher:
    def __init__(self, *, ngram_size, heads_per_ngram, vocab_size, eos_token_id,
                 multipliers, head_sizes, head_offsets):
        self.ngram_size=ngram_size;self.heads_per_ngram=heads_per_ngram
        self.vocab_size=vocab_size;self.eos_token_id=eos_token_id
        self.multipliers=tuple(multipliers);self.head_sizes=tuple(head_sizes)
        self.head_offsets=tuple(head_offsets)
        if (type(ngram_size) is not int or ngram_size<2 or type(heads_per_ngram) is not int
                or heads_per_ngram<1 or type(vocab_size) is not int or vocab_size<1):
            raise ValueError('invalid hash dimensions')
        self._validate_tokens([eos_token_id])
        count=(ngram_size-1)*heads_per_ngram
        if len(self.multipliers)!=ngram_size or len(self.head_sizes)!=count or len(self.head_offsets)!=count:
            raise ValueError('hash buffer shape mismatch')
        if any(type(x) is not int or x<=0 or x*vocab_size>=(1<<63) for x in self.multipliers):
            raise ValueError('multipliers must preserve nonnegative signed64 products')
        offset=0
        for size,start in zip(self.head_sizes,self.head_offsets):
            if type(size) is not int or size<=0 or type(start) is not int or start!=offset:
                raise ValueError('noncontiguous head vocabulary')
            offset+=size
        self.total_rows=offset

    def _validate_tokens(self,tokens):
        if any(type(t) is not int or not 0<=t<self.vocab_size for t in tokens):
            raise ValueError('token outside original vocabulary')

    def hash_tokens(self,tokens,history=None):
        tokens=list(tokens);self._validate_tokens(tokens)
        state=tuple([self.eos_token_id]*(self.ngram_size-1) if history is None else history)
        if len(state)!=self.ngram_size-1:
            raise ValueError('history width mismatch')
        self._validate_tokens(state)
        rows=[]
        for token in tokens:
            previous=[];past_boundary=False
            for prior in reversed(state):
                if prior==self.eos_token_id:past_boundary=True
                previous.append(self.eos_token_id if past_boundary else prior)
            mix=token*self.multipliers[0];ids=[]
            for pos,prior in enumerate(previous,1):
                mix ^= prior*self.multipliers[pos]
                begin=(pos-1)*self.heads_per_ngram
                for head in range(begin,begin+self.heads_per_ngram):
                    ids.append(mix%self.head_sizes[head]+self.head_offsets[head])
            rows.append(ids);state=(*state[1:],token)
        return rows,state
