# nano-vllm Learning Plan & Log

Living document. Checklist tracks what's covered; log entries underneath each
phase capture the actual findings (dated). Check items off as we go — don't
delete finished items, just tick them so the doc stays a record.

**Process:** each phase gets a **reading list + specific questions**. You open
the files, read the functions, and answer the questions yourself (in your own
words, code comments, a scratch file, whatever). Then report back / walk
Claude through your answers — Claude checks, fills gaps, and only then do we
tick the box and move on. Claude will not pre-explain the mechanics before
you've had a pass at them.

**Comment convention:** your notes stay as plain `# ` lines under each
question. Claude's inline replies are `> **🤖 Claude:**` blockquotes, so the
two are visually distinct at a glance — yours are what you wrote, Claude's are
corrections/confirmations layered on top.

**2026-07-03 correction:** Phases 1–3 were originally read/explained by Claude
directly (including running the Phase 3 demo script itself) — that's not the
process above, you didn't drive it. Confirmed 2026-07-03: `sequence.py` and
`scheduler.py` (Phases 1–2) were never actually read in depth either. So the
real order now is **Phase 1 → Phase 2 → Phase 3 → (resume) Phase 4**, all
reopened in reading-list-plus-questions format, in that order, since block_manager
(Phase 3) leans on `Sequence` properties and `prepare_prefill` (Phase 4) leans
on block_manager. Old log entries under 1–3 are Claude's reference notes only
— not verified understanding, don't rely on them.

**2026-07-09 update:** Phase 1 fully walked and checkpointed. Phase 2 mostly
covered — a lot of it happened as live back-and-forth conversation rather than
written-in-file answers, so Claude backfilled blockquote notes under each
question summarizing what was established, and checked off what's actually
verified vs. what's still open. Time estimates added per phase below —
forward-looking pacing estimates only, not a record of actual time spent.
Rough remaining total from here (Phase 2 tail → Phase 9): **~5–7 hours**,
front-loaded toward Phase 4 (CUDA graphs + memory arithmetic is the densest
remaining section) and Phase 6 (model internals).

---

## Phase 0 — Setup & orientation *(Est. 30 min — DONE)*
- [x] Get nano-vllm running end to end on Turing (RTX 2070 SUPER) via SDPA fallback
- [x] Understand why FlashAttention-2 doesn't work on sm75, and what the patch does
- [x] Read HANDOVER.md, confirm environment/run commands

## Phase 1 — `sampling_params.py` + `sequence.py`  *(Est. 45–60 min — DONE, checkpoint walked 2026-07-09)*

**Read:** `sampling_params.py` (8 lines) and `engine/sequence.py` (~85 lines) in full.
Skip `llm_engine.py` for now — you'll want it fresh again after Phase 2 (scheduler) since
`generate()`/`step()` call straight into the scheduler.

<details>
<summary>Claude's prior pass (2026-07-02) — reference only, not verified</summary>

`num_cached_tokens` = tokens whose KV is already resident (from prefix cache or prior
decode steps); `num_scheduled_tokens` = tokens being processed *this* step; a seq is done
with prefill once `num_cached_tokens + num_scheduled_tokens == num_tokens`.
</details>

### 1.1 `sampling_params.py`
- [x] Why does `__post_init__` assert `temperature > 1e-10` instead of `> 0`? What would
      `temperature = 0` mean for sampling, and why might that specific edge case be worth
      explicitly forbidding rather than just letting division happen?
      # doesn't allow temperature = 0 because then it would be division by 0 error. i assume temperature > 1e-10 because we can allow for floats just barely over 0 as a param to reduce randomness
### 1.2 `Sequence.__init__` and the counters
- [x] List every field set in `__init__` and, in one phrase each, what it tracks. Which
      ones start at 0 vs. get a real value immediately — why?
      # self.seq_id - this gets a value because this distinguishes which sequednce the Sequence object is
      # self.status - This gets a value because we initialize it to a default enum. I assume it is waiting because we enqueue it after creating it. it isn't scheduled until we schedule it
      # self.tokens_ids - This gets a real value immediately, this is the raw prompt in list of token ids. we will use this list to create a kv cache
      # last token - this gets a real value immediately because this is the last token in the prompt. We will use this to predict the next token
      # self.num_tokens - this is just the length of the tokenized prompt
      # self.num_prompt_tokens - same as above, but not sure why it is defined again. probably because we will be using it to track if a step has finished
      # self.num_cached_tokens - length of our kv cache essentially for this sequence
      # self.num_scheduled_tokens - length of tokens we scheduled to process in this step. (note: this is only used in prefill)
      > **🤖 Claude:** correction — it's used in decode too. scheduler.py's decode loop sets
      > `seq.num_scheduled_tokens = 1` for every decode step (one new token per seq),
      > then postprocess() resets it to 0 uniformly for both prefill and decode via
      > `seq.num_cached_tokens += seq.num_scheduled_tokens`. It's not prefill-only,
      > it's "how many tokens this step processes" full stop — just always 1 in decode.
      # **User** okay i got it

      # self.is_prefill = True - this is default because at first it is prefill. Then, once all the num_prompt_tokens are cached. (num_cached_tokens >= num_prompt_tokens), then is_prefill = False. I realize why we have two (num_tokens) and (num_prompt_tokens) now sorta. i think num_tokens goes up as we process more and more? but what about num_cached_tokens? oh i get it. because it predicts the next token over and over again till we get the EOS, the answer (response) is still part of this sequence. i'm guessing the final answer is self.token_ids[self.num_prompt_tokens:]
      > **🤖 Claude:** your num_tokens/num_prompt_tokens/completion split is exactly right
      > (confirmed: completion_token_ids property is literally token_ids[num_prompt_tokens:]).
      > But is_prefill's transition mechanism is different than you're guessing — it's not
      > computed by comparing counters anywhere. Grep the whole repo for "is_prefill" and
      > you'll find: (1) it's only ever explicitly SET in 3 places (sequence.py init=True,
      > scheduler.py's decode loop sets it False, scheduler.py's preempt() resets it True),
      > and (2) seq.is_prefill itself is only ever READ in one place — __getstate__ below.
      > It has nothing to do with gating prefill-vs-decode control flow (that's a totally
      > separate bool threaded through schedule()'s return value / model_runner.run()'s
      > argument / utils/context.py's Context.is_prefill — three different "is_prefill"s
      > in this codebase, don't conflate them). So what do you think seq.is_prefill's ONE
      > actual consumer (__getstate__) needs it for? (this sets up 1.5 below)
      # **User** > I don't see getstate being called anywhere though. It does look like though that is_prefill is used to return all of the token ids or just the last token, i also don't get what the decode block does too much. but should we save it later or explain it right now? I don't understand the can_append. Does it return true when prefill is done?
      > **🤖 Claude:** three things here:
      > 1. **`__getstate__`/`__setstate__` call site** — good catch that it's not called
      >    explicitly anywhere. These are Python's *pickle protocol* dunder methods:
      >    `pickle.dumps(obj)` and `pickle.loads(data)` call them implicitly, never by
      >    name. Look at `model_runner.py`'s `write_shm`/`read_shm` — `write_shm` does
      >    `pickle.dumps([method_name, *args])`, and when `call("run", seqs, is_prefill)`
      >    is broadcast to worker ranks, `seqs` (a list of `Sequence` objects) gets pickled
      >    right there, silently invoking `__getstate__` on each one. That's the only
      >    call site in the whole codebase — it exists purely to ship `Sequence` state
      >    across the process boundary to TP worker ranks.
      # **User** > Okay i got it
      > 2. **Your is_prefill read is correct** — confirmed, that's exactly what it gates.
      > 3. **can_append / "does it return true when prefill is done?"** — no, don't
      # **OK**
      >    conflate these two things. `can_append` is a `BlockManager` method (Phase 3
      >    material) called ONLY from the scheduler's *decode* loop, to check "is there a
      >    free KV-cache block available in case this next token crosses into a new
      >    block boundary." It has nothing to do with detecting "prefill is done" — that's
      >    a completely separate check you already found: `seq.num_cached_tokens +
      >    seq.num_scheduled_tokens == seq.num_tokens` in the *prefill* loop (2.1's last
      >    question). Two unrelated mechanisms, easy to conflate since they're both
      >    boolean-ish gates in the scheduler.
      # **OK**
      > 4. **Decode block** — yes, save the deep dive, that's literally Phase 2.2. No
      >    need to preempt it now, just know it exists and sets `is_prefill = False`.
      
      # self.block_table = [] - not sure what this is
      # self.temperature - for randomness, divides the probabilities
      # self.max_tokens - max tokens before we reach EOS
      # self.ignore_eos - ignores eos tokens until we get to max tokens
      

- [x] `token_ids = copy(token_ids)` — why copy instead of storing the reference the caller
      passed in? What could go wrong if you didn't?
      # i'm not sure. i think only something could go wrong if something else would reuse the token_ids, which i dont think will happen. so I'm not too sure. or, if we were to use the initial token id list to store the user prompts, we don't want to mess with it in case we wanted to record the conversation. i think mainly, we want to keep separation of concern
      > **🤖 Claude:** your instinct (aliasing risk) is exactly right, and there IS a concrete
      > mechanism, not just a hypothetical: `append_token()` below does
      > `self.token_ids.append(token_id)` — an in-place mutation. Without the copy,
      > every generated token would get appended directly onto whatever list object the
      > caller passed to `Sequence(...)`, silently corrupting it out from under them
      > (e.g. if they kept a reference to log the original prompt, or passed the same
      > list to build two different Sequences).
      # **User** > Okay i got it
- [x] `num_tokens` vs `num_prompt_tokens` vs `num_cached_tokens` vs `num_scheduled_tokens`
      — these four counters are the crux of the whole engine. For a sequence mid-prefill
      that's had 512 of its 700 prompt tokens processed in a previous step, and is now
      about to process 188 more in this step, what are all four values right before this
      step runs? What are they right after (in `postprocess`)?
      # before this step runs: num_tokens = 700, num_prompt_tokens = 700, num_cached_tokens = 512, num_scheduled_tokens = 188
      # after postprocess: num_tokens = 700, num_prompt_tokens = 700, num_cached_cached_tokens = 700, num_scheduled_tokens = 0
      > **🤖 Claude:** "before" is exactly right. "After" has a real bug in it though — reread
      > postprocess() in scheduler.py line by line for this exact case:
      > ```
      > seq.num_cached_tokens += seq.num_scheduled_tokens   # 512+188 = 700
      > seq.num_scheduled_tokens = 0
      > if is_prefill and seq.num_cached_tokens < seq.num_tokens: continue
      >     # 700 < 700 is FALSE, so this does NOT continue —
      >     # execution falls through to the next line:
      > seq.append_token(token_id)
      > ```
      > So num_tokens becomes 701, not 700, in this same postprocess call — the moment
      > prefill finishes is the SAME step where the first generated token gets appended,
      > there's no separate "finished prefill, now wait a step" gap. This is exactly what
      > 2.4 below asks you to trace — good to have hit it here first though.
      # **User** Okay i got it

### 1.3 Derived properties
- [x] `num_blocks` and `last_block_num_tokens` — hand-compute both for `num_tokens=700`,
      `block_size=256`. Then explain in words what `last_block_num_tokens` represents when
      the sequence is NOT block-aligned.
      # for computing num_blocks = 3, why not have it so that its return -(self.num_tokens // -self.block_size) basically ceiling of the division
      # last_block_num_tokens is 188. this means that the last block for this sequence will have 188 tokens
      > **🤖 Claude:** both numbers correct (3 and 188). And your alternate ceiling-division
      > idiom is genuinely valid: `-(-a // b) == (a + b - 1) // b` for positive a, b —
      > `-(700 // -256) = -(-3) = 3`, checks out. Same result, just a different (relies on
      > Python's floor-toward-negative-infinity `//` semantics) way to write "ceiling
      > division." Pure style choice, no functional difference — the codebase's
      > `(num_tokens + block_size - 1) // block_size` form is more common because it
      > doesn't require reasoning about negative-number floor division to trust it.
      # **User** Okay i got it
      
      # not block-aligned im not sure what that means 
      > **🤖 Claude:** "block-aligned" means `num_tokens` is an exact multiple of
      > `block_size` (e.g. 512, 768, 1024 for block_size=256) — every block, including
      > the last, would be completely full, so `last_block_num_tokens` would just equal
      > `block_size`. Our example (700) is NOT aligned — 700/256 isn't a whole number —
      > so the last block only has 188 tokens sitting in it (the other 68 slots in that
      > block are allocated but unused until more tokens arrive). That partial fill is
      > exactly why the last block is always excluded from `can_allocate`'s hash check
      > (Phase 3) — it isn't "complete" yet, its content can still change.
      # **OK**

- [x] `block(i)` — what does it return, and why does `can_allocate`/`allocate` in
      block_manager call this instead of just indexing `token_ids` directly? (You don't
      need to have read block_manager yet — just explain what `block(i)` itself computes.)
      # block(i) returns all of the tokens at block i
      # i think it calls block(i) rather than indexing directly because it's self contained within the sequence. will need more of an explanation here
      > **🤖 Claude:** correct on both counts. Precisely: `block(i)` returns
      > `token_ids[i*block_size : (i+1)*block_size]` — a slice of up to `block_size`
      > tokens (fewer, for the final partial block). On "why not index directly":
      > two concrete reasons beyond just "self-contained" —
      > 1. `block_manager` would otherwise have to duplicate the `i*block_size` /
      >    `(i+1)*block_size` arithmetic itself every call site; keeping it in one method
      >    means if `block_size` bookkeeping ever changed, only `Sequence.block()` changes.
      > 2. You get the `assert 0 <= i < self.num_blocks` bounds check for free — any bug
      >    in block_manager's loop bounds (e.g. off-by-one in `can_allocate`'s
      >    `range(seq.num_blocks - 1)`) would immediately throw here instead of silently
      >    returning a wrong/empty slice.

      # **OK**

### 1.4 `append_token` and finish/status transitions
- [x] Where is `SequenceStatus` (WAITING/RUNNING/FINISHED) actually mutated? (grep the
      whole codebase, not just this file — it's set from outside `Sequence`.) Why does
      `Sequence` itself not decide when it's finished?

      # sequence itself just stores the state of the sequence. it shouldn't manage itself because that's the scheduler's job
      # sequence status is mutated in the scheduler. This makes sense because the role of the scheduler is to manage the sequence state
      # sequence status states:
      # running - set when the prefill is done, and needs to do decode
      # waiting - set when the prefill has not started yet, but needs to do prefill
      # finished - set when eos or max tokens is reached
      > **🤖 Claude:** good instinct on the separation-of-concerns reasoning, and all 3
      > transitions are right. One small correction: it's not *exclusively* the
      > scheduler — the very first value, WAITING, is set inside `Sequence.__init__`
      > itself (`self.status = SequenceStatus.WAITING`), at construction time, before the
      > scheduler ever sees it. So: **initial value** is set by `Sequence` itself (as
      > part of being born), and every **transition after that** (WAITING→RUNNING,
      > RUNNING→FINISHED, RUNNING→WAITING on preempt) is exclusively the scheduler's job.
      > Also worth flagging for Phase 2.3: WAITING isn't only "hasn't started prefill" —
      > it's also where a *preempted* sequence goes back to (from RUNNING), meaning it
      > has to redo its prefill from scratch. Same status, two different real histories.
      # **OK** 

### 1.5 `__getstate__`/`__setstate__`
- [x] This is for cross-process (TP) pickling. Explain the ternary: `last_state =
      self.last_token if not self.is_prefill else self.token_ids`. Why send the *whole*
      token list during prefill but only the *last* token during decode? What would break
      (or just be wasteful) if it always sent the whole list?
      # for decode, we only care about the last token because the previous tokens are already cached. it would be a waste
      # for prefill, we need the whole token list because we haven't processed all the tokens yet. But also, i'm wondering if we can just send token_ids[num_cached_tokens:]
      > **🤖 Claude:** decode reasoning: correct. Prefill reasoning: your instinct that
      > the full list is more than strictly necessary is actually right — good catch.
      > `prepare_prefill` only ever reads `seq[start:end]` where `start =
      > seq.num_cached_tokens`, so a worker really only needs `token_ids[num_cached_tokens:]`
      > for the step it's about to run. Sending the whole list isn't *wrong*, just not
      > minimal. The likely reason it's not optimized: prefill only sends this payload a
      > handful of times per sequence (once, or a few times if chunked across a long
      > prompt) — the overhead is bounded and small. Decode is where this actually
      > matters: it sends *some* payload every single generated token (could be hundreds
      > of steps), so if it sent the full growing `token_ids` each time like prefill
      > does, that's O(n) bytes every step → O(n²) total over a generation. That
      > frequency difference, not a correctness difference, is why decode got the
      > "only send last_token" optimization and prefill didn't bother.
- [x] Why doesn't `__getstate__` include `temperature`/`max_tokens`/`ignore_eos`? (Think
      about which rank actually needs those — tie back to what you confirmed earlier about
      `self.sampler` and `if self.rank == 0` in `model_runner.py`.)
      # these are applied whenever the next token is generated, during decode. it's not really a state of the sequence but rather just a config. 
      > **🤖 Claude:** the config-vs-state framing is right, but the question was
      > steering at something more specific: WHICH rank actually touches these fields at
      > all? Trace it: `temperature` is consumed by `prepare_sample()`, which is gated
      > `if self.rank == 0` in `model_runner.run()`; `max_tokens`/`ignore_eos` are
      > consumed by the finish check in `scheduler.postprocess()`, which only ever runs
      > in the main process (never shipped to workers at all — `Scheduler` isn't
      > distributed). So worker ranks (rank>0), which reconstruct a `Sequence` via
      > `__setstate__` purely to run `prepare_prefill`/`prepare_decode`/the model
      > forward pass, never touch these 3 fields — full stop. It's not a stylistic
      > omission, it's dead weight on the receiving end: there is no code path on a
      > worker rank that would ever read `seq.temperature`.
      # **OK**
      
### Checkpoint
Walk Claude through 1.1–1.5 before moving to Phase 2.

## Phase 2 — `engine/scheduler.py`  *(Est. 60–90 min — IN PROGRESS, ~80% covered as of 2026-07-09)*

**Read:** `engine/scheduler.py` in full (~93 lines).

<details>
<summary>Claude's prior pass (2026-07-02) — reference only, not verified</summary>

Prefill-first batching loop, chunked prefill (only first seq in a batch may be chunked),
continuous batching admission; decode loop with preemption when `block_manager.can_append`
fails; `postprocess()` hashes newly-completed blocks, appends sampled token, checks
EOS/max_tokens, deallocates on finish. Preemption evicts the *most recently added* running
seq (LIFO via `self.running.pop()`), not the seq that actually needs the block.
</details>

### 2.1 `schedule()` — prefill loop
- [x] Walk the `while self.waiting and len(scheduled_seqs) < self.max_num_seqs:` loop
      line by line. What does `remaining` track, and what causes the loop to `break`
      entirely vs. just stop adding *this* seq?
      **User** - The remaining tracks the budget left in this batch. the loop breaks entirely when 
      there are no more budget for tokens left, or if there are no more kv blocks left to allocate
      we stop adding this prefix when we have scheduled all of the tokens left in this sequence
      > **🤖 Claude:** both `break` conditions (budget exhausted, `can_allocate` returns
      > -1) are exactly right. One refinement: there's actually no code path where the loop
      > "just stops adding this seq but keeps going with others" within the same call —
      > every exit from this loop is a full `break`. Proved this later in conversation:
      > whenever the chunked-prefill branch fires (`remaining < num_tokens and
      > scheduled_seqs`), `num_scheduled_tokens = min(num_tokens, remaining)` always equals
      > `remaining` exactly, draining the budget to 0 and forcing the very next iteration to
      > `break` at the top anyway. So "fully schedule this seq and continue" or "break
      > entirely" are the only two outcomes per call — never a partial skip.
- [x] `if not seq.block_table:` branches into calling `block_manager.can_allocate` —
      what does an empty `block_table` mean about this seq's state (new arrival vs.
      continuing a chunked prefill)?
      **User** - empty block table means that this is the first time this sequence is going through a batch. and no blocks has been allocated for this sequence yet
      non empty block_table means that we already allocated the blocks for this sequence, and it is chunked prefill
      (this means that we have previously already ran a prefill run on this )
      > **🤖 Claude:** correct on both. Confirmed later too: `allocate()` always reserves
      > blocks for the *entire* prompt in one shot (based on `seq.num_blocks`, computed
      > from the full `num_tokens` set at `Sequence.__init__`) — not incrementally per
      > chunk — so a non-empty `block_table` really does mean "already fully allocated,
      > just resuming token-by-token processing," never "partially allocated."
- [x] The comment-equivalent logic `if remaining < num_tokens and scheduled_seqs: break`
      — explain in your own words what "only allow chunked prefill for the first seq"
      means and why the check is specifically `and scheduled_seqs` (i.e. only applies if
      something's already been scheduled this call).
      > **🤖 Claude — covered live in conversation, 2026-07-09:** `scheduled_seqs` is empty
      > only for the very first candidate examined in a call — that's the one exemption
      > that lets a sequence whose `num_tokens` alone exceeds the entire budget still get
      > admitted (chunked), specifically to guarantee liveness (without it, a too-big
      > prompt would `break` every single call, forever, deadlocking the engine). Once ≥1
      > seq is already scheduled this call, that liveness guarantee is already satisfied,
      > so a 2nd/3rd candidate that doesn't fit just gets `break`'d entirely and retried
      > next call with a fresh full budget — chunking it too would only fragment the step
      > for no completion benefit. Also found: nano-vllm's "only first seq chunks" is a
      > simplification vs. real vLLM (which packs partial chunks from multiple seqs to
      > avoid wasting leftover budget) — traced a concrete example (950 + 1050 token seqs,
      > budget 1000) where full chunking finishes in 2 steps vs. nano-vllm's 3, confirming
      > it does cost some throughput for scheduler simplicity.
- [x] `if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:` — what
      real-world condition does this detect, and what two things happen to the seq when
      it's true?
      > **🤖 Claude — covered live in conversation, 2026-07-09:** detects "this step's work
      > brings the sequence's cached-token count all the way up to its full prompt length"
      > — i.e. prefill just finished (including finishing exactly via a chunk this step).
      > The two things: `seq.status = SequenceStatus.RUNNING`, and it moves from
      > `self.waiting` to `self.running` (`popleft()` + `append()`). Runs regardless of
      > whether this step chunked or fully processed the seq — purely about the
      > *cumulative* cached-token total.

### 2.2 `schedule()` — decode loop
- [ ] Why does the decode loop only run `if scheduled_seqs` is empty from the prefill
      loop above (look at the `if scheduled_seqs: return ...` right before it)? What does
      that imply about whether prefill and decode ever happen in the same `schedule()`
      call?
- [x] Walk the `while not self.block_manager.can_append(seq):` inner loop and the
      `for...else` construct. What is `self.preempt(self.running.pop())` actually evicting,
      and why pop from the *end* of `running` rather than the *front*? What happens in the
      degenerate case where `seq` itself is the only running sequence and still can't get a
      block?
      > **🤖 Claude — covered live in conversation, 2026-07-09:** `self.running.pop()` (right
      > end) evicts the most-recently-scheduled running sequence first — LIFO, sacrificing
      > younger sequences before ones with more invested work. Each preemption evicts the
      > victim's *entire* `block_table` (`deallocate()` has no partial mode) — the "just
      > enough" behavior comes from the `while` re-checking `can_append` after each full
      > eviction, not from evicting partial amounts. Degenerate case: when `self.running`
      > is empty (nobody left to sacrifice), it preempts `seq` itself and `break`s out of
      > the inner loop (skipping the `else`), so `seq` doesn't get scheduled this step
      > either — sent back to `waiting` like any other victim.

### 2.3 `preempt`
- [x] Three things happen in `preempt`. What state does a preempted sequence lose, and
      where does it go (front or back of `waiting`)? Why is losing its blocks safe/correct
      given what you know about `hash_to_block_id` from block_manager (even without having
      read block_manager in depth yet — just reason about whether recomputation vs.
      correctness is at stake)?
      **User** - i think losing its blocks is safe/correct because we only erase the blocks
      allocated to this sequence → refined to: `deallocate()` only touches
      `seq.block_table`/`seq.num_cached_tokens`, never `seq.token_ids`, so we just lose
      time (redo prefill), not correctness, since a forward pass is deterministic given the
      same tokens.
      > **🤖 Claude:** exactly right, confirmed 2026-07-09. Also nailed the follow-up: it
      > goes to the *front* of `waiting` (`appendleft`), and freed blocks aren't wiped —
      > `_deallocate_block` doesn't clear `block.hash`/`block.token_ids`, so a freed block
      > is still findable via `hash_to_block_id` until some other `allocate()` call claims
      > that exact `block_id`. Rescheduling the victim ASAP (front of queue) minimizes that
      > window, giving it a real shot at a partial/full prefix-cache hit on its own prior
      > work rather than a guaranteed full recompute.

### 2.4 `postprocess()`
- [ ] Order of operations matters here: `hash_blocks` is called *before*
      `seq.num_cached_tokens += seq.num_scheduled_tokens`. Why does the hashing need to
      happen first — what would be wrong about the block boundaries if you incremented
      `num_cached_tokens` first?
- [x] `if is_prefill and seq.num_cached_tokens < seq.num_tokens: continue` — what real
      scenario does this `continue` skip past (i.e., don't append a sampled token yet)?
      **User** (from inline scheduler.py comment) — "if its prefill and we haven't cached
      every token, we skip this sequence in postprocess."
      > **🤖 Claude:** correct — this skips appending a token when a chunked-prefill step
      > didn't finish the whole prompt yet; there's no "next token" to sample/append until
      > prefill is actually complete.
- [ ] The finish condition checks `seq.num_completion_tokens == seq.max_tokens` — go
      back to `sequence.py`'s `num_completion_tokens` property and confirm you know
      exactly which tokens that counts.

### Checkpoint
2.1 (all 4), 2.2's second bullet, and 2.3 are walked and verified (2026-07-09).
Remaining before Phase 2 is fully closed: 2.2's first bullet, and 2.4's first and
third bullets.

## Phase 3 — BlockManager: paged KV cache + prefix caching  *(Est. 30–45 min remaining)*

**2026-07-09 note:** a surprising amount of this phase already happened as a side
effect of Phase 2 conversation — `can_allocate`/`allocate` mechanics, `ref_count`
sharing across sequences (cache-hit blocks vs. private decode-time blocks),
`_allocate_block`/`_deallocate_block` and the `free_block_ids` FIFO ordering,
`hash_blocks` timing/purpose, and `can_append`'s block-boundary check were all
covered in depth already. What's left is mostly formalizing it against 3.1–3.6's
specific questions below (some will go fast) and actually running the demo script
yourself (3.6) — probably closer to 30 min than the original 60–90 estimate.

**Read:** `engine/block_manager.py` in full (~120 lines). Have `engine/sequence.py`
and `engine/scheduler.py` open alongside since block_manager is called from there.

<details>
<summary>Claude's prior pass (2026-07-02) — reference only, don't treat as verified until you've derived it yourself</summary>

`Block` struct: `block_id`, `ref_count`, `hash`, `token_ids`. `compute_hash` is xxhash64,
chained (`hash(block_i) = xxhash(prev_hash_bytes + block_i_tokens_bytes)`) — a
rolling/Merkle-style chain, not a per-block content hash. `can_allocate` walks blocks
`0..num_blocks-2` (last block always excluded), stops at first hash miss. `allocate`
reuses cached blocks (bumping ref_count, or reclaiming from free list) then allocates
fresh blocks for the rest. `hash_blocks` (called from `postprocess`, not `schedule`)
only hashes the range that just became "complete" this step. Ran a demo: two prompts
sharing a 601-token prefix as two separate `llm.generate()` calls — second prompt hit
`num_cached_blocks=2`, `block_table=[0,1,3]` (reused blocks 0,1, block 2 excluded as A's
trailing partial block). Demo script (still usable, run it yourself now):
`/tmp/claude-0/-root-nano-vllm/be71e56e-fd78-495c-bb1a-80e16dde13f5/scratchpad/prefix_demo.py`
</details>

### 3.1 `Block` + `compute_hash`
- [ ] What do the 4 fields of `Block` represent? What does `ref_count` count,
      physically — shared across how many what?
- [ ] `compute_hash(cls, token_ids, prefix=-1)` folds a `prefix` hash into the digest
      before hashing this block's tokens. Why? Concretely: could two *different*
      sequences ever get the same hash for block index 2 if their block-2 token content
      is identical but their block-0/block-1 content differs? Why does that matter for
      correctness of prefix-cache reuse?

### 3.2 `can_allocate`
- [ ] It loops `for i in range(seq.num_blocks - 1)` — not `range(seq.num_blocks)`.
      Why is the last block always excluded? What would go wrong (give a concrete
      failure) if it checked the last block too?
- [ ] Inside the loop: `if block_id in self.used_block_ids: num_new_blocks -= 1`.
      Explain the difference between a cache-hit block that's `used_block_ids` vs. one
      that's only sitting in `free_block_ids`, and why that distinction changes how many
      *new* blocks still need to be allocated.
- [ ] Hand-trace: a seq has `num_blocks=4`. The hash chain matches at blocks 0 and 1,
      then misses at block 2. What does `can_allocate` return? What does it NOT check
      about block 3?

### 3.3 `allocate`
- [ ] For a reused (cache-hit) block, trace both branches: one where the block is
      already in `used_block_ids`, one where it's only in `free_block_ids`. What happens
      to `ref_count` and the free/used sets in each case, and why do they differ?
- [ ] What happens to `seq.num_cached_tokens` at the end of `allocate`, and how does
      that value get used later (you saw this already in Phase 4's `prepare_prefill` —
      `start = seq.num_cached_tokens`)?

### 3.4 `hash_blocks` — the trickiest one
- [ ] Why is this called from `postprocess()` (scheduler.py) rather than from
      `schedule()`? What would be wrong/premature about hashing a block the moment it's
      scheduled, before the model has actually run?
- [ ] Hand-trace: `seq.num_cached_tokens=0`, `seq.num_scheduled_tokens=300`,
      `block_size=256`. Compute `start` and `end`. Which block index(es) get hashed?
      Does the partially-filled second block (44 tokens) get hashed on this call? Why
      or why not — is that a bug or intentional given the floor division?

### 3.5 `deallocate` / `can_append` / `may_append`
- [ ] `deallocate`: confirm the ref-counting — when does a block actually return to
      `free_block_ids`?
- [ ] `can_append`'s return is `len(free_block_ids) >= (len(seq) % self.block_size ==
      1)`. This is in the "open questions" list below — resolve it now: at exactly what
      point in a growing decode sequence does `len(seq) % block_size == 1` become true,
      and why does that mean "we need a new block right now"?

### 3.6 Hands-on demo — run it yourself
- [ ] Run `/tmp/claude-0/-root-nano-vllm/be71e56e-fd78-495c-bb1a-80e16dde13f5/scratchpad/prefix_demo.py`
      yourself (it's already instrumented to print `can_allocate`/`allocate` calls).
      Before running, predict: what will `num_cached_blocks` be for prompt A? For prompt
      B? What will `block_table` look like for each? Then run it and check your
      predictions against the output, using 3.1–3.5 to explain *why* the numbers came
      out that way (don't just match them).

### Checkpoint
Walk Claude through your answers to 3.1–3.6 before we resume Phase 4.

## Phase 4 — GPU bridge: `model_runner.py`  *(Est. 90–120 min — PAUSED, resume after Phase 1, 2 & 3 checkpoints)*

You've already made a first pass at `__init__`/`exit`/`loop`/`read_shm` via inline
`# Claude:` comments in the file — that stands, no need to redo it. 4.1–4.5 (the
substantive `prepare_prefill`/`prepare_decode`/`allocate_kv_cache`/CUDA-graph exercises)
are on hold until Phase 3 is walked, since 4.1 leans directly on `can_allocate`/`allocate`.

**Read:** `engine/model_runner.py` (all of it — it's ~260 lines) and
`utils/context.py` (short). Don't read `layers/attention.py` yet, that's Phase 5.

### 4.1 `prepare_prefill` (lines ~130–171)
- [ ] For a given `seq`, what do local vars `start` and `end` represent? Why is
      `start = seq.num_cached_tokens` instead of always `0`? Tie this back to what
      you learned about `num_cached_tokens` in Phase 1/3.
- [ ] `cu_seqlens_q` vs `cu_seqlens_k` are appended-to identically most of the time.
      Find the one line where they'd diverge for a given seq, and explain in words
      when `seqlen_q != seqlen_k` happens. What does the `if cu_seqlens_k[-1] >
      cu_seqlens_q[-1]:` check right after the loop decide, and why does prefix-cache
      prefill need `block_tables` but a normal cold prefill doesn't?
- [ ] Walk through the `slot_mapping` inner loop (`start_block`, `end_block`,
      `slot_start`, `slot_end`) by hand for a concrete example: a seq with
      `num_cached_tokens=512`, `num_scheduled_tokens=100`, `block_size=256`,
      `block_table=[7, 12, 3]`. What slot indices end up in `slot_mapping`? (Write
      out the actual numbers — this is the exercise.)
- [ ] Why does the loop `continue` (skip slot_mapping entirely) `if not
      seq.block_table`? Which caller hits this path? (Check `warmup_model`.)

### 4.2 `prepare_decode` (lines ~173–189)
- [ ] Compare to `prepare_prefill` — why is this function so much shorter? What's
      true about every seq's `seqlen_q`/`seqlen_k` here that removes the need for
      `cu_seqlens` entirely?
- [ ] `slot_mapping.append(seq.block_table[-1] * self.block_size +
      seq.last_block_num_tokens - 1)` — explain each term. What single KV-cache
      slot is this computing, and why is it always exactly one slot per seq?
- [ ] `seq.last_block_num_tokens` is a `Sequence` property from Phase 1 — go check
      its definition again and confirm you can explain why `-1` is needed at the end.

### 4.3 `allocate_kv_cache` (lines ~104–122)
- [ ] Explain the memory-arithmetic line:
      `num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak +
      current) // block_bytes`. Why does it subtract `peak` and add back `current`
      — what double-counting is that correcting for? (Hint: think about what
      `torch.cuda.memory_stats()` peak/current mean relative to `used` from
      `mem_get_info`, and why `warmup_model()` is called right before this.)
- [ ] What are the 6 dimensions of `self.kv_cache` (`torch.empty(2,
      num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)`)?
      What does the leading `2` represent? Where do `module.k_cache`/`v_cache` get
      wired up, and to which module type (go find it — you don't need to read
      `qwen3.py` in depth yet, just locate the attribute)?

### 4.4 CUDA graphs — `capture_cudagraph` / `run_model` (lines ~196–258)
- [ ] `run_model` only uses a captured graph when NOT prefill, NOT enforce_eager,
      and batch size ≤ 512. Why can't prefill ever use a CUDA graph (think about
      what varies between prefill calls — shapes? the whole point of graphs?).
- [ ] In `capture_cudagraph`, buffers like `slot_mapping`, `block_tables` etc. are
      allocated ONCE at max size, and `run_model` copies real data into a *slice* of
      them (`graph_vars["slot_mapping"][:bs] = context.slot_mapping`) before
      `graph.replay()`. Why must the buffers be reused in place rather than passing
      fresh tensors each call? What CUDA graph constraint does this satisfy?
      This directly explains why the Turing patch had to force `enforce_eager=True`
      — go reread the one-line reason in `ModelRunner.__init__` (`self.enforce_eager
      = config.enforce_eager or not flash_attn_supported()`) and connect it to what
      you just learned about static buffers.

### 4.5 `utils/context.py`
- [ ] How does `set_context()` in `prepare_prefill`/`prepare_decode` get read by
      the attention layer without `context` being passed as an explicit argument
      through `model.forward()`? What mechanism is this (module-level global?
      thread-local? contextvar?), and why is `reset_context()` called at the end of
      every `run()`?

### Checkpoint
When you've worked through 4.1–4.5, tell Claude your answers (or paste your
notes) and we'll verify together, then check off the boxes and log findings
below.

## Phase 5 — KV-cache write path *(Est. 45–60 min)*
- [ ] `store_kvcache` triton kernel — how `slot_mapping` addresses the paged cache
- [ ] Connect back to `layers/attention.py`: the three modes (causal prefill, prefix-cache
      prefill with explicit bottom-right mask, paged decode with block_table gather) —
      already read once during the Turing patch work; revisit with fresh block_manager
      knowledge to see how `block_table` / `slot_mapping` feed the SDPA fallback

## Phase 6 — Model internals: `models/qwen3.py` *(Est. 60–90 min)*
- [ ] Overall model graph (embed → transformer blocks → norm → lm_head)
- [ ] RoPE (`layers/rotary_embedding.py` or wherever it lives)
- [ ] GQA (grouped-query attention) — how num_kv_heads < num_heads is handled
- [ ] RMSNorm (`layers/layernorm.py`)
- [ ] Activation (`layers/activation.py`) — SwiGLU or similar
- [ ] `layers/embed_head.py` — embedding + LM head (tied weights?)
- [ ] `sampler.py` — the Gumbel-trick sampling + per-seq temperature broadcast (mentioned
      as already partially understood from Phase 1 read of llm_engine, revisit in depth)
- [ ] `utils/loader.py` — weight loading/mapping from HF checkpoint

## Phase 7 — Tensor parallelism *(Est. 30–45 min — conceptual only, single-GPU box)*
- [ ] `layers/linear.py` — column/row-parallel linear layers, all-reduce points
- [ ] How `tensor_parallel_size > 1` spawns processes in `llm_engine.py` (`mp.spawn`, events)
      and how `Sequence.__getstate__`/`__setstate__` fit in (cross-process seq state)

## Phase 8 — Turing patch deep dive (revisit with full context) *(Est. 30 min)*
- [x] High-level: `flash_attn_supported()`, SDPA fallback reproducing 3 attention modes
- [ ] Line-by-line reread of the SDPA fallback now that block_manager/model_runner are
      understood — check the explicit bottom-right mask construction for prefix-cache
      prefill, and the K/V gather-from-block_table for decode

## Phase 9 — Stretch goals (optional) *(Est. open-ended)*
- [ ] Benchmark: measure actual prefill/decode speedup from prefix caching (time prompt B
      with vs. without a shared-prefix warm cache)
- [ ] Try `kvcache_block_size` other than 256 (needs multiple-of-256 assert — inspect why)
- [ ] Read upstream vLLM's equivalent block manager to see what nano-vllm simplified away
- [ ] Try tensor_parallel_size if a second GPU/process ever becomes available (likely N/A
      on this single-GPU box, but worth understanding the code path even unexercised)

---

## Open questions / things to come back to
- Why does `config.py` assert `kvcache_block_size % 256 == 0` specifically (multiple of
  256, not just "positive")? Look for a triton kernel tiling constraint.
- `can_append`'s return is `len(free_block_ids) >= (len(seq) % block_size == 1)` — the
  boolean-as-int trick for "do we need a new block for the next appended token." Confirm
  this is exactly the point at which a decode step crosses a block boundary.
