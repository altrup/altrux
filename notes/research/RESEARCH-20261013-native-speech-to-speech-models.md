# Research - 2026-10-13: native speech-to-speech models

## Summary

- A native speech-to-speech model works on discrete or continuous audio frames, not on ASR text. The front end that the field has standardised on is a causal convolutional (plus small transformer) codec at **12.5 Hz**. Most of these codecs make the first codebook semantic, and use residual codebooks for acoustics.
- There are two output families. **Multi-codebook RVQ with a per-frame depth module** (Moshi, CSM, Qwen3-Omni, NVIDIA VoiceChat) needs no separate vocoder model. **Single semantic codebook with a flow-matching decoder** (GLM-4-Voice, Kimi-Audio, Step-Audio 2) does.
- Every competitive system starts from a text LLM and keeps a text stream inside it: an inner monologue, an interleaved text track, or a Thinker-Talker split. Textless spoken LMs learn language about three orders of magnitude more slowly. Speaking costs knowledge: Moshi drops from 75.0 to 62.3 on spoken Llama Questions against its own text backbone.
- Full duplex is native in only a few open models (Moshi and its derivatives, Hertz-dev, MiniCPM-o 4.5, NVIDIA VoiceChat). Most others are turn-based with VAD.
- For altrux: a 12.5 Hz frame clock gives one recurrent step per 80 ms. That suits a Mamba2 state, which does not grow with stream length. The open problems are what a dream and a KL target mean for acoustic tokens. Locally, only codec and sub-2B model inference is realistic on 8 GB.

Scope: front end (waveform to tokens), back end (tokens to waveform), the model families that use them, training recipes, evaluation, and what this means for the [self-teaching end goal](../discussion/DISCUSSION-20261010-self-teaching-end-goal.md). This is background only. It does not change the LAMA-CKL direction.

## 1. The pipeline in one picture

```text
waveform 24 kHz
  -> causal conv encoder (strided, e.g. x1920) -> latent at 12.5 Hz
  -> quantiser: RVQ (Q codebooks) or single VQ / FSQ
  -> token frame t: [c0 (semantic), c1..c(Q-1) (acoustic)]
  -> sequence model: one backbone step per frame, plus a small depth module over c0..c(Q-1)
  -> token frame t+1
  -> codec decoder (conv) or flow matching + vocoder -> waveform
```

Two numbers set the cost:

1. **Frame rate.** This is the backbone steps per second. It went from 50-75 Hz (EnCodec, SoundStream, HuBERT units) to 12.5 Hz (Mimi, GLM-4-Voice, Qwen3-Omni) and 6.25 Hz (Qwen3.5-Omni input).
2. **Codebooks per frame.** These are either flattened into the sequence (costly), or predicted by a small per-frame module, so the backbone takes one step per frame.

This is the same choice as in the [desktop tokenization note](RESEARCH-20260825-desktop-multimodal-action-state-tokenization.md) section 1. A continuous encoder can be projected into the LM (the LLaVA pattern; here Whisper-encoder inputs), or the modality can be put into one discrete vocabulary (the Chameleon pattern; here codec tokens). Speech output forces a discrete or generative decoder, so most systems use a continuous input with a discrete output, or discrete on both sides.

## 2. Neural audio codecs and speech tokenizers

### 2.1 Comparison

Frame rate is in frames per second. Tokens/s counts all codebooks. "Causal" means the paper states a causal or streaming encoder. A blank or "n/s" means not stated in the source that was read.

| Codec | Source, date | SR | Frame rate | Codebooks x size | Tokens/s | kbps | Encoder | Semantic content | Weights |
|---|---|---|---|---|---|---|---|---|---|
| SoundStream | [2107.03312](https://arxiv.org/abs/2107.03312), Jul 2021 | 24 kHz | 75 Hz | RVQ, e.g. 8 x 1024 at 6 kbps | 600 at 6 kbps | 3-18 | conv, causal, streaming | none | no |
| EnCodec | [2210.13438](https://arxiv.org/abs/2210.13438), Oct 2022 | 24 kHz (causal), 48 kHz (non-causal) | 75 Hz (24 kHz) | RVQ, 2-32 x 1024 | 150-2400 | 1.5-24 | conv + LSTM, causal at 24 kHz | none | yes ([repo](https://github.com/facebookresearch/encodec)) |
| DAC | [2306.06546](https://arxiv.org/abs/2306.06546), Jun 2023 | 44.1 kHz | 86 Hz | RVQ, 9 x 1024 | ~774 | 8 | fully conv, non-causal (not stated explicitly) | none | yes, MIT ([repo](https://github.com/descriptinc/descript-audio-codec)) |
| SpeechTokenizer | [2308.16692](https://arxiv.org/abs/2308.16692), Aug 2023 | 16 kHz | 50 Hz | RVQ, 8 x 1024 | 400 | 4.0 | conv + BiLSTM, non-causal | HuBERT distilled into RVQ-1 | yes |
| Mimi (Moshi) | [2410.00037](https://arxiv.org/abs/2410.00037), Sep 2024 | 24 kHz | 12.5 Hz | split RVQ, 8 x 2048 (up to 32) | 100 | 1.1 | conv + 8-layer causal transformers; 80 ms frame, streaming | WavLM distilled into codebook 1 | yes, CC-BY-4.0 ([card](https://huggingface.co/kyutai/mimi)) |
| WavTokenizer | [2408.16532](https://arxiv.org/abs/2408.16532), Aug 2024 | 24 kHz | 40 or 75 Hz | 1 x 4096 | 40 or 75 | 0.48 or 0.9 | conv + attention, iSTFT decoder | none explicit | yes |
| X-Codec | [2408.17175](https://arxiv.org/abs/2408.17175), Aug 2024 | 16 kHz | 50 Hz | RVQ x 1024 | 50 per codebook | n/s | conv | HuBERT/WavLM features fused before RVQ, semantic loss | yes |
| BigCodec | [2409.05377](https://arxiv.org/abs/2409.05377), Sep 2024 | 16 kHz | 80 Hz | 1 x 8192 | 80 | 1.04 | conv + LSTM, 159M params | none | code released |
| Stable Codec (TAAE) | [2411.19842](https://arxiv.org/abs/2411.19842), Nov 2024 | 16 kHz | 25 Hz | FSQ, 1 or 2 x 15625 | 25 or 50 | 0.4 or 0.7 | ~950M transformer, non-causal | none | yes, Stability community licence |
| GLM-4-Voice tokenizer | [2412.02612](https://arxiv.org/abs/2412.02612), Dec 2024 | n/s | 12.5 Hz | 1 x n/s | 12.5 | 0.175 | Whisper-large-v3 encoder + VQ; causal conv, block-causal attention | ASR-supervised | yes |
| X-codec2 (Llasa) | [2502.04128](https://arxiv.org/abs/2502.04128), Feb 2025 | 16 kHz | 50 Hz | FSQ, 1 x 65536 | 50 | ~0.8 | w2v-BERT + conv encoder, transformer decoder | semantic encoder fused | yes, CC-BY-NC-4.0 ([card](https://huggingface.co/HKUSTAudio/xcodec2)) |
| FocalCodec | [2502.04465](https://arxiv.org/abs/2502.04465), Feb 2025 | 16 kHz | 50 / 25 / 12.5 Hz | 1 x 8192 (binary) | 50 / 25 / 12.5 | 0.65 / 0.33 / 0.16 | focal modulation; streaming variant | by design | code released |
| Qwen-TTS-Tokenizer-12Hz | [2601.15621](https://arxiv.org/abs/2601.15621), Jan 2026 | n/s | 12.5 Hz | 1 semantic + 15 RVQ, x 2048 | 200 | ~2.2 | fully causal, no look-ahead | WavLM into codebook 1 (Mimi recipe) | yes, Apache-2.0 |
| OmniCodec | [2603.20638](https://arxiv.org/abs/2603.20638), Mar 2026 | 16 kHz | 12.5 or 6.25 Hz | RVQ up to 32 x 2048 | 100-400 | 1.1-4.4 | purely causal; Qwen3-Omni AuT encoder as semantic branch | yes | announced, not released at paper time |

Semantic-only tokens (no acoustic reconstruction on their own):

| Tokens | Source | Frame rate | Vocabulary | Notes |
|---|---|---|---|---|
| w2v-BERT k-means (AudioLM) | [2209.03143](https://arxiv.org/abs/2209.03143), Sep 2022 | 25 Hz | 1024 | 250 bps; non-causal encoder |
| HuBERT k-means (dGSLM) | [2203.16502](https://arxiv.org/abs/2203.16502), Mar 2022 | 50 Hz | 500 | HiFi-GAN unit vocoder |
| mHuBERT units (SpeechGPT) | [2305.11000](https://arxiv.org/abs/2305.11000), May 2023 | n/s | K n/s | adjacent duplicates removed |
| CosyVoice S3 | [2407.05407](https://arxiv.org/abs/2407.05407), Jul 2024 | n/s in the text that was read | 4096 | VQ inside an ASR (SenseVoice) encoder |

Values marked "~" are arithmetic from the stated rates, not quoted values.

### 2.2 What the table shows

- **Frame rate fell by about 6x in two years.** The 12.5 Hz rate is now the default for dialogue models: Mimi, GLM-4-Voice, Kimi-Audio, Baichuan-Audio, Qwen3-Omni, Step-Audio 2 (input) and NVIDIA VoiceChat. The [DualCodec](https://arxiv.org/abs/2505.13000) paper claims to be "the first open-source 12.5 Hz low frame-rate codec" for its data, and it credits Mimi as the first open-weight one. Qwen3.5-Omni ([2604.15804](https://arxiv.org/abs/2604.15804), Apr 2026) uses 6.25 Hz on its input encoder.
- **Semantic and acoustic content conflict.** Mimi states the trade directly. Its split-RVQ design (a separate semantic quantiser) improves MUSHRA from 57.8 to 64.0, while ABX (phonetic discriminability, lower is better) degrades from 6.5% to 8.1%. SpeechTokenizer shows that RVQ-1 alone carries the content: WER is 20.02 on RVQ-1 vs 5.04 on all eight levels. X-Codec reports that adding semantic features cuts VALL-E-style TTS WER from 6.37 (EnCodec) to 4.07.
- **Two routes to "semantic first".** One route distils an SSL model (WavLM, HuBERT) into codebook 1 (Mimi, SpeechTokenizer, Qwen-12Hz). The other puts a VQ bottleneck inside an ASR encoder (CosyVoice S3, GLM-4-Voice, Qwen-25Hz). The ASR route gives very low bitrate (GLM-4-Voice at 175 bps). It discards paralinguistics, so it needs a strong generative decoder (flow matching) to restore voice.
- **Single codebook vs RVQ.** A single codebook (WavTokenizer, BigCodec, X-codec2, FocalCodec) gives one token per frame and a plain LM. RVQ at 12.5 Hz gives 8-32 tokens per frame, and needs a depth module or delayed parallel heads (section 3.2).
- **Causality is not universal.** Only SoundStream, EnCodec-24k, Mimi, GLM-4-Voice's tokenizer, Qwen-12Hz and OmniCodec state a causal or streaming encoder in the sources that were read. Full duplex needs a causal codec on both sides.
- **Objective metrics are weak.** The Moshi paper reports "a concerning lack of correlation between objective and subjective audio quality metrics" (VisQOL vs MUSHRA on Mimi).

## 3. How the sequence model produces audio tokens

### 3.1 Taxonomy of text use

| Text role | Mechanism | Examples |
|---|---|---|
| None (textless) | audio tokens only | AudioLM, dGSLM, SyncLLM, Hertz-dev |
| Chain of modality | full text answer first, then speech | SpeechGPT |
| Text-first, speech decoder on hidden states | LLM writes text; a separate speech head reads its states | LLaMA-Omni (1/2), Freeze-Omni, Mini-Omni, MiniCPM-o 4.5 |
| Fixed-ratio interleave | alternate text and audio chunks in one stream | GLM-4-Voice (13 text : 26 audio), Step-Audio 2, SpiRit-LM (word-level) |
| Thinker-Talker | text LLM ("Thinker") plus an audio LM ("Talker") that reads Thinker hidden states | Qwen2.5-Omni, Qwen3-Omni, Qwen3.5-Omni |
| Inner monologue | time-aligned text tokens as a prefix to each audio frame, inside one multistream step | Moshi, Hibiki, Hibiki-Zero, PersonaPlex |
| Parallel heads | one backbone step emits a text token and audio tokens from separate heads | Kimi-Audio, NVIDIA VoiceChat |

### 3.2 Multi-codebook schemes

- **Flattening.** Put all Q codebooks of a frame in sequence. This costs Q backbone steps per frame. It is used only at low Q or offline.
- **Delayed parallel streams** ([MusicGen](https://arxiv.org/abs/2306.05284) style). Predict all codebooks in one step from Q heads, with codebook k delayed by k steps. Mini-Omni uses this with SNAC (7 layers) plus text: "text-first delay parallel decoding" with eight heads.
- **Depth transformer (RQ-Transformer).** A large temporal transformer takes one step per frame. A small depth transformer then predicts codebooks 0..Q-1 autoregressively inside the frame. In Moshi the depth transformer has 6 layers at dimension 1024. Each 80 ms step generates 17 tokens (1 text + 8 Moshi audio + 8 for the user stream's slot) "instead of 16 without" inner monologue. [Sesame CSM](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) uses the same split: the backbone models codebook 0, and a 100M-300M decoder models the rest. Qwen3-Omni predicts codebook 0 with the backbone head and the residual codebooks with an 80M multi-token-prediction (MTP) module. NVIDIA VoiceChat predicts 31 RVQ levels per 80 ms frame.
- **Acoustic delay.** Moshi delays the acoustic codebooks one or two frames after the semantic one: "we pretrain Moshi with an acoustic delay of 2 and finetune it with an acoustic delay of 1, for a theoretical latency of 160ms". Kyutai's [Delayed Streams Modeling](https://arxiv.org/abs/2509.08753) (Sep 2025) generalises this. One decoder-only multistream model becomes ASR or TTS according to which stream is delayed: "ASR corresponds to the text stream being delayed, while the opposite gives a text-to-speech (TTS) model."
- **Continuous frames.** Hertz-dev models a continuous VAE latent (a 32-dim latent per 125 ms frame, 8 Hz) instead of codes ([blog](https://si.inc/posts/hertz-dev), [repo](https://github.com/Standard-Intelligence/hertz-dev)). Kyutai's [CALM](https://arxiv.org/abs/2509.06926) (Sep 2025) has an MLP generate the next continuous frame from the backbone state. This removes the codebook-depth problem, but it needs a sampling head instead of a softmax. It was not read in depth for this note.

### 3.3 Full duplex

There are three designs:

1. **Two synchronous streams in one step.** Moshi models the user stream and its own stream in parallel at each 12.5 Hz step, so it can speak and listen at once with no turn segmentation. This holds for its derivatives: [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA, Jan 2026) and Hibiki. dGSLM uses two towers with cross-attention on Fisher two-channel audio. Hertz-dev adds two projection heads in post-training.
2. **Time-chunk interleaving.** [SyncLLM](https://arxiv.org/abs/2409.15594) (Meta, Sep 2024) alternates 160-240 ms chunks of each speaker on a wall clock, from a Llama-3-8B initialisation. [MiniCPM-o 4.5](https://arxiv.org/abs/2604.27393) (Apr 2026) uses 1 s time windows ("Omni-Flow").
3. **Half-duplex core with a state predictor.** Freeze-Omni predicts a listen/speak/interrupt state per chunk. Mini-Omni2 uses "command-based interruption". Commercial APIs gate turns with VAD and support barge-in (section 4.3).

Latency figures from the sources:

| System | Latency |
|---|---|
| Moshi | 160 ms theoretical, 200 ms in practice on an L4 |
| Hertz-dev | 120 ms real-world on one RTX 4090 |
| GPT-4o | 232 ms minimum, 320 ms average |
| Qwen3-Omni | 234 ms first packet, cold start, audio input |
| LLaMA-Omni | 236 ms (paper; the README says 226) |
| LLaMA-Omni2 | 550-660 ms |
| Freeze-Omni | ~1.2 s including network |

## 4. Model survey

### 4.1 Research and open full-duplex lineage

| Model | Source, date | Input / output tokens | Streams, text use | Duplex | Backbone, data | Weights |
|---|---|---|---|---|---|---|
| dGSLM | [2203.16502](https://arxiv.org/abs/2203.16502), Mar 2022 | HuBERT 500 units, 50 Hz | two towers + cross-attention; textless | yes | Fisher 2000 h | n/s |
| AudioLM | [2209.03143](https://arxiv.org/abs/2209.03143), Sep 2022 | w2v-BERT 25 Hz + SoundStream 50 Hz | three hierarchical stages; textless; continuation only | n/a | 0.3B per stage; Libri-Light 60k h | no |
| SpeechGPT | [2305.11000](https://arxiv.org/abs/2305.11000), May 2023 | mHuBERT units | one stream; chain of modality | no | LLaMA-13B; 60k h LibriLight adaptation | yes |
| SpiRit-LM | [2402.05755](https://arxiv.org/abs/2402.05755), Feb 2024 | HuBERT 25 Hz (+ pitch 12.5 Hz, style 1 Hz) | one stream, word-level interleave | no | Llama-2 7B; 460k h speech | yes, FAIR non-commercial ([repo](https://github.com/facebookresearch/spiritlm)) |
| Moshi | [2410.00037](https://arxiv.org/abs/2410.00037), Sep 2024 | Mimi 12.5 Hz x 8, both sides | user + model audio + inner monologue; depth transformer | yes, native | Helium 7B (2.1T text tokens); 7M h audio, Fisher 2000 h, >20k h synthetic dialogue | yes, CC-BY-4.0 ([repo](https://github.com/kyutai-labs/moshi)) |
| SyncLLM | [2409.15594](https://arxiv.org/abs/2409.15594), Sep 2024 | HuBERT units + timing tokens | interleaved 160-240 ms chunks; textless at inference | yes | Llama-3 8B; 212k h synthetic + 2k h Fisher | none found |
| Mini-Omni / Mini-Omni2 | [2408.16725](https://arxiv.org/abs/2408.16725), Aug 2024; [2410.11190](https://arxiv.org/abs/2410.11190), Oct 2024 | Whisper-small in; SNAC 7 layers out | delayed parallel heads, text first | no / command interrupt | Qwen2-0.5B; ~8k h ASR + VoiceAssistant-400K | yes, MIT (Mini-Omni) |
| LLaMA-Omni / LLaMA-Omni2 | [2409.06666](https://arxiv.org/abs/2409.06666), Sep 2024; [2505.02625](https://arxiv.org/abs/2505.02625), May 2025 | Whisper-large-v3 in; HuBERT units with CTC (v1) / 25 Hz 6561-token CosyVoice-2-style + flow matching (v2) | text first, speech decoder on hidden states | no | Llama-3.1-8B / Qwen2.5 0.5-14B; 200K samples | yes (v1 non-commercial) |
| Freeze-Omni | [2411.00774](https://arxiv.org/abs/2411.00774), Nov 2024 | 12.5 Hz streaming encoder in; TiCodec 1 x 1024 at 40 Hz out | frozen LLM, text first | chunk state predictor | Qwen2-7B frozen; 110k h ASR, 3k h TTS | yes |
| Hertz-dev | [blog](https://si.inc/posts/hertz-dev), Nov 2024 | continuous VAE latent, 8 Hz, 16 kHz | audio-only; two heads for duplex | yes | 8.5B, init from a 2T-token text LM; hours n/s | yes |
| Sesame CSM | [blog](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice), Feb 2025 | Mimi 12.5 Hz | text + audio context in, audio out; backbone + audio decoder | n/a (speech generator, not dialogue) | 1B/3B/8B backbones; ~1M h audio | 1B only, Apache-2.0 ([card](https://huggingface.co/sesame/csm-1b)) |
| Hibiki | [2502.03382](https://arxiv.org/abs/2502.03382), Feb 2025 | Mimi 12.5 Hz, up to 16 codebooks | source + target audio + inner monologue | simultaneous translation | 2.7B; ~40k h per language | yes ([repo](https://github.com/kyutai-labs/hibiki)) |
| Hibiki-Zero | [2602.11072](https://arxiv.org/abs/2602.11072), Feb 2026 | as Hibiki | as Hibiki; GRPO for latency, no word-aligned data | simultaneous translation | 3B, Helium-2B init; 40k h per source language | yes ([repo](https://github.com/kyutai-labs/hibiki-zero)) |
| PersonaPlex | [2602.06053](https://arxiv.org/abs/2602.06053), Jan 2026 | Mimi (Moshi-based) | Moshi + text role prompt + voice prompt | yes | 7B; ~2.2k h synthetic + Fisher | yes, NVIDIA open model licence ([card](https://huggingface.co/nvidia/personaplex-7b-v1)) |
| NVIDIA VoiceChat | [2609.21967](https://arxiv.org/abs/2609.21967), Sep 2026 | 600M streaming encoder in (12.5 Hz); 31-level RVQ at 12.5 Hz, 22 kHz out | parallel text + function-call streams, RNN-T user transcript, streaming TTS decoder | yes, with tool calls | Nemotron-Nano-9B-v2; hours n/s | checkpoint stated on HF; licence n/s |

Kyutai [Unmute](https://github.com/kyutai-labs/unmute) is a cascade (DSM STT, then any text LLM, then DSM TTS), not a native model. It is in this note because its STT and TTS ([DSM repo](https://github.com/kyutai-labs/delayed-streams-modeling)) are the low-latency streaming components of the same family: stt-1b-en_fr has a 0.5 s delay and stt-2.6b-en a 2.5 s delay.

### 4.2 Industry omni models

| Model | Source, date | Audio in | Audio out | Text use | Duplex | Size, data | Weights |
|---|---|---|---|---|---|---|---|
| GLM-4-Voice | [2412.02612](https://arxiv.org/abs/2412.02612), Dec 2024 | own 12.5 Hz single-codebook tokens | same tokens, then flow matching + HiFi-GAN | 13 text : 26 audio interleave ("Streaming Thoughts") | no | GLM-4-9B; 1T tokens, 700k h speech | yes ([card](https://huggingface.co/THUDM/glm-4-voice-9b)) |
| Qwen2.5-Omni | [2503.20215](https://arxiv.org/abs/2503.20215), Mar 2025 | Whisper-init encoder, ~40 ms per frame | qwen-tts-tokenizer, then DiT flow matching + BigVGAN | Thinker-Talker | no | 7B; 300B audio-related tokens in stage 2 | yes |
| Qwen3-Omni | [2509.17765](https://arxiv.org/abs/2509.17765), Sep 2025 | AuT encoder (650M, 20M h supervised), 12.5 Hz | multi-codebook RVQ: backbone predicts codebook 0, 80M MTP predicts the rest, 200M causal ConvNet | Thinker (30B-A3B MoE) - Talker (3B-A0.3B) | no | ~2T tokens stage 2 (0.77T audio) | yes, Apache-2.0 ([repo](https://github.com/QwenLM/Qwen3-Omni)) |
| Qwen3.5-Omni | [2604.15804](https://arxiv.org/abs/2604.15804), Apr 2026 | AuT (40M h), 6.25 Hz | RVQ + MTP + ConvNet; ARIA rate-adaptive interleave | Thinker-Talker | no | "hundreds of billions" of parameters; >100M h audio-visual | open status unconfirmed |
| Step-Audio | [2502.11946](https://arxiv.org/abs/2502.11946), Feb 2025 | dual tokens: 16.7 Hz x 1024 linguistic + 25 Hz x 4096 semantic, 2:3 interleave | 3B speech LM + flow matching + vocoder | interleaved | no | 130B; 1.1T audio tokens (~7.3M h) | Chat 130B and TTS-3B only |
| Step-Audio 2 | [2507.16632](https://arxiv.org/abs/2507.16632), Jul 2025 | frozen 25 Hz encoder, adaptor to 12.5 Hz | CosyVoice 2 tokens (6.6K added to the vocabulary) + flow matching | fixed-ratio interleave; CoT + RL, tool calls | no | 680B text tokens + 8M h audio | n/s |
| Kimi-Audio | [2504.18425](https://arxiv.org/abs/2504.18425), Apr 2025 | 12.5 Hz semantic tokens + Whisper features downsampled to 12.5 Hz | 12.5 Hz semantic tokens + flow matching + BigVGAN | parallel text and audio heads | no | Qwen2.5-7B init; >13M h audio | n/s |
| Baichuan-Audio | [2502.17239](https://arxiv.org/abs/2502.17239), Feb 2025 | 8-level RVQ at 12.5 Hz ({8K, 4K, 2K, 1K x5}) on a Whisper encoder | same + 3-layer depth transformer, flow matching | text-guided alternation | no | 7B; 887k h audio | yes ([repo](https://github.com/baichuan-inc/Baichuan-Audio)) |
| VITA-Audio | [2505.03739](https://arxiv.org/abs/2505.03739), May 2025 | GLM-4-Voice tokenizer | GLM-4-Voice decoder; 10 audio tokens per forward pass (MCTP) | interleaved | no | Qwen2.5-7B; ~200k h open data | yes |
| MiniCPM-o 4.5 | [2604.27393](https://arxiv.org/abs/2604.27393), Apr 2026 | Whisper-medium, 10 tokens/s after the projector | 0.3B speech decoder emitting S3 tokens + streaming flow matching | LLM (Qwen3-8B) emits text only | yes (Omni-Flow, 1 s windows) | ~9B | n/s |

### 4.3 Closed models (public facts only)

- **GPT-4o** ([system card, 2410.21276](https://arxiv.org/abs/2410.21276), Aug 2024). The card says it is "trained end-to-end across text, vision, and audio, meaning that all inputs and outputs are processed by the same neural network". It responds "in as little as 232 milliseconds, with an average of 320 milliseconds". The tokenizer, codec, size and data are not disclosed.
- **gpt-realtime** ([announcement](https://openai.com/blog/introducing-gpt-realtime/), 2025-08-28). This is the GA speech-to-speech model of the Realtime API. Secondary reports ([InfoQ](https://infoq.com/news/2025/09/openai-gpt-realtime)) give 82.8% on Big Bench Audio, vs 65.6% for the Dec 2024 model. The [Realtime docs](https://developers.openai.com/api/docs/guides/realtime-conversations) use 24 kHz PCM or G.711, VAD on by default (including `semantic_vad`), and cancel the response on barge-in. That is turn-gated, not a public full-duplex model. A later [gpt-realtime-2](https://developers.openai.com/api/docs/models/gpt-realtime-2) model page exists (128K context); its launch date (around May 2026) is from secondary sources only.
- **Gemini native audio.** The Live API takes 16 kHz PCM and returns 24 kHz PCM over a WebSocket. [Gemini 3.1 Flash Live](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-flash-live/) was released on 2026-03-26. The [Gemini 3.8 Audio model card](https://deepmind.google/models/model-cards/gemini-3-8-audio/) (2026-09-15) says the Live models are "based on Gemini 3 Pro", with a 128K input context and audio and text out. It gives no architecture or latency figures. Google discloses no tokenizer.

## 5. Training recipes

### 5.1 Text initialisation wins

- **Scratch audio pretraining is slow.** Cuervo and Marxer ([2404.00685](https://arxiv.org/abs/2404.00685), Mar 2024) find that "linguistic performance scales up to three orders of magnitude more slowly in SLMs relative to LLMs". They also find that synthetic story speech (sTinyStories) beats audiobooks at every scale, and that coarser tokenization hurts. These figures come from the HTML page through a fetch summary and were not checked against the PDF table.
- **Text initialisation helps, and scale helps more.** TWIST ([2305.13009](https://arxiv.org/abs/2305.13009), May 2023) trained on ~150k h of 25 Hz HuBERT units. At 1.3B, text initialisation beats cold initialisation by only ~0.5 points on sWUGGY and sBLIMP. Moving to 7B and 13B text initialisations adds more. Spoken StoryCloze stays near chance: 55.4 at 13B, vs 79.9 for humans.
- **All production systems start from a text LLM.** Hertz-dev is the only audio-only model in the survey, and it also starts from a text LM. SpeechSSM (section 5.4) starts from RecurrentGemma.

### 5.2 Keep text in the loop during audio training

- **Moshi keeps half its batches text-only during audio pretraining.** It uses a separate optimiser state for those batches. Without them, spoken TriviaQA drops from 22.8 to 18.3: "a warm start from Helium is not sufficient to retain the knowledge of the original text model". MMLU still falls from 54.3 (Helium) to 49.7 (Moshi).
- **Interleaving at scale.** Zeng et al. ([2411.17607](https://arxiv.org/abs/2411.17607), Nov 2024) make 600B tokens of synthetic interleaved speech-text from text corpora. A text-to-token model does this, so no audio synthesis is needed. Spoken QA rises from 13% (Moshi) to 31%. GLM-4-Voice is built on this recipe. SpiRit-LM interleaves at word level on real aligned speech (~111k h).
- **Replay against forgetting.** Hsiao et al. ([2505.17496](https://arxiv.org/abs/2505.17496), May 2025) compare model merging, LoRA-scale discounting and experience replay across staged ASR, TTS and spoken-QA training. Replay preserves text knowledge best. The abstract was read; the numbers were not.

### 5.3 Synthetic dialogue

Real two-channel dialogue is scarce. Fisher (2000 h of 8 kHz phone calls) is the shared base: dGSLM, Moshi, SyncLLM and PersonaPlex all use it. Everything above that is TTS-rendered LLM dialogue:

- **Moshi.** Over 20k h of synthetic dialogue from a Helium fine-tune and its own multistream TTS, with one consistent voice from a single actor in 70+ styles. Moshi reports that text instruct data with URLs, bullets and lists renders badly as speech.
- **SyncLLM.** 212k h of synthetic dialogue.
- **PersonaPlex.** ~1840 h of customer-service dialogue and 410 h of QA.

### 5.4 The modality gap

- **Input gap.** VoiceBench ([2410.17196](https://arxiv.org/abs/2410.17196), Oct 2024) gave the same prompts as text and as speech. The ASR-pipeline baseline loses 4.37 points. End-to-end models such as VITA lose more than 35 points, and most are near random on spoken multiple choice. LLaMA-Omni's speech tuning cost its text tasks more than 11 points. These are 2024 models; Qwen2.5-Omni later claims speech-input MMLU and GSM8K "comparable" to text input.
- **Output gap.** Moshi's own table compares Helium written answers with Moshi spoken answers: Llama Questions 75.0 vs 62.3, TriviaQA 56.4 vs 22.8. This mixes several factors (audio training, oral-style fine-tuning, the speech channel). No clean "same model, written vs spoken answer" number was found. [2609.15313](https://arxiv.org/abs/2609.15313) (Sep 2026, output-mode gap, on-policy distillation) is the lead to read; it was found by search and not read.
- **Mechanism.** [2510.12116](https://arxiv.org/abs/2510.12116) (Oct 2025) finds that speech and text hidden states align in direction but diverge in magnitude, and that token-level alignment predicts the gap. Read through a summary only.
- **What mitigates it:**
  - **Inner monologue.** Without it, Moshi spoken QA is 9.2 / 21.0 / 7.3 (Web Q / Llama Q / TriviaQA); with it, 26.6 / 62.3 / 22.8. The cost is one extra token per step.
  - **Thinker-Talker.** The text path is never asked to produce audio, so text reasoning stays intact by construction.
  - **Text replay** during audio training.
  - **RL alignment.** TARS ([2601.05543](https://arxiv.org/abs/2601.05543), 2026) uses GRPO with representation- and behaviour-alignment rewards. Its numbers were not extracted.

### 5.5 Recurrent backbones

- **SpeechSSM** ([2412.18603](https://arxiv.org/abs/2412.18603), Dec 2024) calls itself "the first state-space spoken LM". It is Griffin (gated linear recurrence plus local attention), initialised from RecurrentGemma 2B/9B, on 25 Hz USM semantic tokens, with SoundStorm and SoundStream for acoustics. It trains on 4-minute sequences and generates up to 16 minutes in constant memory. It is textless, yet it has the best transcript perplexity against larger transformer baselines. It is weak on sWUGGY, which the authors blame on the large vocabulary. It is the closest precedent to a Mamba2 speech LM.
- **Other recurrent speech work.**
  - [DuplexMamba](https://arxiv.org/abs/2502.11123) (Feb 2025): a Mamba LM with speech in and text out, and duplex through a duplicated state. It is not a native audio generator. Read from the abstract only.
  - [Speech Slytherin](https://arxiv.org/abs/2407.09732) (Jul 2024): Mamba for codec TTS (VALL-M).
  - [RWKVTTS](https://arxiv.org/abs/2504.03289) (Apr 2025): RWKV-7 in codec-token TTS.
- **No Mamba-based native speech-to-speech dialogue model was found.** The search was limited.

## 6. Evaluation

| Benchmark | Source | Measures | Note |
|---|---|---|---|
| sWUGGY, sBLIMP (ZeroSpeech 2021) | [2011.11588](https://arxiv.org/abs/2011.11588) | lexical and syntactic preference by likelihood, textless | Moshi finds that they do not track common-sense ability; SpeechSSM finds that they are noisy with large vocabularies |
| sStoryCloze / tStoryCloze | TWIST [2305.13009](https://arxiv.org/abs/2305.13009) | spoken story-ending choice | sSC stays near chance even at 13B |
| Spoken Web Q / Llama Q / audio TriviaQA | Spectron [2305.15255](https://arxiv.org/abs/2305.15255); Moshi | spoken factual QA | the de facto knowledge probe for spoken LMs |
| VoiceBench | [2410.17196](https://arxiv.org/abs/2410.17196), Oct 2024 | spoken instructions: AlpacaEval, CommonEval, SD-QA, OpenBookQA, MMSU, IFEval, AdvBench, with speaker, noise and content variation | text vs speech pairing exposes the input gap |
| Big Bench Audio | [Artificial Analysis](https://artificialanalysis.ai/methodology/speech-to-speech-benchmarking) | 1000 BBH questions as audio (4 categories), audio in and out, judge-scored | the main public speech-to-speech reasoning leaderboard; no paper |
| VoxEval | [2501.04962](https://arxiv.org/abs/2501.04962), Jan 2025 | knowledge QA with spoken question and spoken answer, robustness, spoken math | weak reasoning, sensitive to audio conditions |
| URO-Bench | [2502.17810](https://arxiv.org/abs/2502.17810), Feb 2025 | speech-to-speech understanding, reasoning, oral conversation, multilingual, paralinguistic | open models show "catastrophic forgetting" against their LLM backbones |
| Full-Duplex-Bench v1 / v1.5 | [2503.04721](https://arxiv.org/abs/2503.04721), Mar 2025; [2507.23159](https://arxiv.org/abs/2507.23159), Jul 2025 | pauses, backchannels, turn-taking, interruption; overlap handling and stop latency | v1.5 finds "responsive" vs "floor-holding" strategies |
| Talking Turns | [2503.01174](https://arxiv.org/abs/2503.01174), Mar 2025 | turn-taking judged by a model trained on human dialogue | systems interrupt too much and rarely backchannel |
| Output quality | Seed-TTS-eval WER, UTMOS, MUSHRA, speaker similarity | intelligibility and naturalness of generated speech | e.g. Qwen2.5-Omni WER 1.42 / 2.33 / 6.54 (zh / en / hard) |

## 7. Relevance to altrux

This section argues from the [self-teaching end goal](../discussion/DISCUSSION-20261010-self-teaching-end-goal.md). Points marked **(speculation)** are not supported by any source above.

### 7.1 Why audio is worth knowing about here

The end goal is a continual learner for computer use: pixels in, fixed action grammar out, reward anchored on the user and the environment. Voice is not in that goal. The voice literature is still the most mature case of a single sequence model that consumes one continuous sensory stream and produces another continuous stream in real time, with a frozen tokenizer on both sides. The design questions match those open in the end-goal note and the [desktop note](RESEARCH-20260825-desktop-multimodal-action-state-tokenization.md):

- **Frozen vs live encoder** (end goal, open question 3). Every surveyed system freezes its codec after training it once. The field never consolidates against a moving tokenizer. That supports the "frozen" side of the end-goal argument. It is evidence of practice, not a test of the question.
- **Cycle consistency** (end goal, section 4). A codec is a learned encode-decode round trip with a reconstruction loss. Moshi's own output stream is re-read as input at the next step. For audio the "environment" (the codec decoder plus the ear) is cheap and differentiable. For pixels it needs a learned forward model. **(speculation)** Audio is the cheap rehearsal of the screen-action loop.
- **Text inside the loop.** Section 5.4 is the strongest finding: models that drop the text stream lose most of their knowledge. The analogy for altrux **(speculation)**: a pixels-in model will need a text-like channel for facts and plans, as the desktop note's fixed action grammar keeps `TYPE(text)` in the language tokenizer.

### 7.2 A codec front end on a Mamba2 backbone

| Quantity | Value | Source |
|---|---|---|
| Mamba2 2.7B recurrent state | 64 layers x 80 heads x 64 headdim x 128 d_state ~ 41.9M values per sequence, constant in length (plus conv state) | `models/mamba2_2_7b/model.py` docstring; HF config |
| Text tokens per second of speech | ~3-4 BPE tokens/s at ~150 words/min | estimate, not sourced |
| Backbone steps per second, 12.5 Hz codec with depth module | 12.5 | Mimi |
| Tokens per second, Mimi x 8, flattened | 100 | Mimi |
| One hour of one audio stream | 45,000 backbone steps (12.5 Hz) | arithmetic |

Consequences:

- **A recurrent state fits a frame clock.** A transformer dialogue model's KV cache grows with every 80 ms frame. A Mamba2 state does not. SpeechSSM's 16-minute constant-memory generation is the evidence. Full duplex as "two streams summed into one step" (the Moshi pattern) maps directly onto one recurrent step per frame.
- **The content density per step is about 3-4x lower than text** at 12.5 Hz with a depth module. With flattened RVQ it is about 30x lower. The [binding-capacity note](../discussion/DISCUSSION-20260804-binding-capacity-and-null-rerun.md) measured roughly 3-4 bindings in the 780M state over text. **(speculation)** On audio the same state spends most of its capacity on acoustic detail, so facts per second of speech would fall further. A semantic-only first codebook, or an inner-monologue text stream, is where facts would live.
- **The backbone needs a depth module.** Mamba2 would be the temporal model. A small transformer or MLP (Moshi: 6 layers; Qwen3-Omni: 80M MTP) predicts the codebooks within the frame. That is new code in `models/`, not a reuse of the current LM head.

### 7.3 Continual learning on audio streams **(speculation throughout)**

- **Wake.** Listening to a stream is the analogue of reading the evidence document. The LAMA-CKL protocol needs a frame-level target. With an inner monologue the text stream gives the existing scorer (LAMA facts in text) unchanged. Without it, scoring must go through spoken QA (Llama Q / TriviaQA style), which needs ASR and adds noise.
- **Dreams.** A dream would be the model generating audio tokens from an EOC-like prompt. Two risks follow from the third-box collapse ([DISCUSSION-20261013](../discussion/DISCUSSION-20261013-lama-ckl-third-box-debrief.md)). First, self-generated acoustic tokens degrade toward low-entropy loops (silence, repeated frames) even more easily than text repeats a frame. Second, a KL over all eight codebooks is dominated by acoustic codebooks that carry voice and channel, not content. A distillation target restricted to the text stream plus semantic codebook 0 is the first candidate.
- **Reward from the user.** In a duplex stream the user's backchannels, interruptions and prosody arrive at frame resolution. They are a dense, external, graded signal of the kind the end-goal note wants for its reward token. Full-Duplex-Bench and Talking Turns already score these behaviours. Whether they correlate with "the answer was good" is unknown.
- **Data.** Real two-channel dialogue is scarce (Fisher, 2000 h), and every system relies on TTS-rendered synthetic dialogue. A continual learner on synthetic speech risks consolidating TTS artefacts.

### 7.4 Open questions

1. Can a Mamba2 temporal model with a depth module match a transformer on spoken QA at equal parameters and frame rate? SpeechSSM suggests yes for continuation, but not for dialogue.
2. Should altrux use a 12.5 Hz RVQ codec (Mimi, Qwen-12Hz) or a single semantic codebook with a flow-matching decoder? The first keeps a frame-synchronous duplex loop. The second gives a smaller per-step target, but needs a heavy decoder.
3. Does dream distillation on the semantic codebook preserve facts while the acoustic codebooks drift?
4. Is the inner monologue required, or only helpful? The end goal wants no fine-tuning phases. A text stream that must stay aligned to audio is a structural constraint to decide once.

### 7.5 What is cheap to try locally (8 GB AMD, inference only)

| Item | Fits | Use |
|---|---|---|
| Mimi encode and decode ([kyutai/mimi](https://huggingface.co/kyutai/mimi), via the `moshi` or `transformers` package) | yes; ~100M-class model (size not checked) | tokenise own speech; measure codebook entropy and per-codebook content; listen to reconstructions at 1-8 codebooks |
| Kyutai STT 1B / TTS 1.6B ([DSM](https://github.com/kyutai-labs/delayed-streams-modeling)) | likely in bf16 (not tested) | streaming baseline; source of aligned text-audio streams |
| Sesame CSM-1B | likely (not tested) | Mimi-token generation from text; dream-like audio continuation |
| Mini-Omni (0.5B) | yes | smallest end-to-end speech-in, speech-out loop |
| Moshi 7B | no in bf16 (the README asks for 24 GB); the int8 PyTorch build is marked experimental | not local |
| Hertz-dev 8.5B, Qwen3-Omni 30B, GLM-4-Voice 9B | no | not local |

ROCm caveat: these are plain PyTorch (conv plus transformer) and should not touch the broken `causal-conv1d` and Triton kernels described in `AGENTS.md`. None has been run on this box. The cheapest useful probe: encode an hour of LibriSpeech with Mimi, then measure (a) the token budget per second at Q = 1, 2, 8, and (b) ASR WER on the decode at each Q. That is the facts-per-step input to 7.2. It needs no training.

## Claims not verified against a primary source

- Codec causality for DAC, WavTokenizer, X-Codec, X-codec2 and BigCodec. The GLM-4-Voice codebook size. The SpeechGPT unit count. The CosyVoice S3 frame rate. The Qwen2.5-Omni and Qwen3-Omni output codebook counts.
- The Cuervo and Marxer details beyond the headline sentence (fetch summary only).
- Papers found only through search summaries: [2510.12116](https://arxiv.org/abs/2510.12116), [2609.15313](https://arxiv.org/abs/2609.15313), DuplexMamba, Spirit LM's speech-vs-text numbers, URO-Bench, VoxEval and Talking Turns findings.
- The gpt-realtime Big Bench Audio score (secondary reports), the gpt-realtime-2 launch date, Qwen3.5-Omni open weights, and the NVIDIA VoiceChat licence and latency.
- Hertz-dev training hours and licence.
- All "fits" claims in 7.5 except Moshi's 24 GB requirement.

## Sources checked

The primary papers above were read as PDF text, except where marked. Also read: the GitHub READMEs for Moshi, Hibiki, Hibiki-Zero, DSM, Unmute, Hertz-dev, Sesame CSM, Mini-Omni, LLaMA-Omni, Freeze-Omni and Spirit LM; the Hugging Face cards for Mimi, CSM-1B, X-codec2 and PersonaPlex; the OpenAI Realtime docs; and the Gemini 3.8 Audio model card. Repository starting points: the [notes index](../README.md), the [desktop tokenization note](RESEARCH-20260825-desktop-multimodal-action-state-tokenization.md), and the [end-goal note](../discussion/DISCUSSION-20261010-self-teaching-end-goal.md).
