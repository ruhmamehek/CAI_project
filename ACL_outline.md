# 🧠 **Low-Resource and Multilingual Conversational AI: Challenges, Methods, and Future Directions**

*(Revised Logical Flow — No Overlap Version)*

---

### **1. Abstract (150–200 words)**

Concise overview of motivation, scope, contributions, and key findings.
*(Write last.)*

---

### **2. Introduction & Motivation (~1 page)**

* Define *low-resource* and *multilingual* in dialogue AI.
* Quantify global linguistic imbalance (95%+ underrepresented).
* Impact on accessibility: assistants, education, civic tech.
* Research evolution: monolingual → cross-lingual → multilingual → instruction-tuned LLMs.
* **Contributions:** synthesis, taxonomy, evaluation comparison, cultural/safety insights.

---

### **3. Scope & Terminology Box (short boxed section)**

Clarify:

* *Multilingual* vs *cross-lingual* vs *code-switched*.
* Data vs compute scarcity.
* Scope limited to text-based conversational systems (dialogue, chatbots, assistants).

---

### **4. Related Surveys & Historical Context (0.5 page)**

Summarize prior reviews (multilingual NLP, low-resource MT, code-switching) and highlight novelty:

> Focus on dialogue-centric multilinguality, instruction-tuned LLMs, and evaluation fairness.

---

### **5. Foundations of Multilingual Representation (~1 page)** -- Stefanie

*(Technical groundwork only — no fairness overlap.)*

* **Tokenization & Scripts:** subword, byte-level, and morphology issues in non-Latin languages.
* **Pretraining Paradigms:**

  * Encoder: mBERT, XLM-R.
  * Encoder-decoder: mBART, mT5.
  * Byte/char: ByT5.
* **Learning Objectives:** MLM, TLM, contrastive alignment.
* **Comparative Analysis:** encoder vs seq2seq vs byte-level.
* **Figure:** taxonomy diagram of multilingual representation families.
* End note: “These representations underpin all multilingual dialogue models explored next.”

---

### **6. Multilingual & Low-Resource Dialogue Datasets (~1.5 pages)** -- Stefanie

* **Benchmark Overview Table** (MTOP, MASSIVE, XPersona, XDialog, GLUECoS, XTREME).
* **Dataset Creation Pipelines:** translation-based, community-sourced, synthetic augmentation.
* **Data Quality Challenges:** annotation consistency, script diversity.
* **Ethical Dimensions:** participatory collection, indigenous rights, licensing, cultural representation.

---

### **7. Cross-Lingual Adaptation & Transfer Techniques (~1.5 pages)** -- Aditya

*(New merged + streamlined section replacing 7 & 8 overlap.)*

* **Fine-Tuning Strategies:** zero-shot, translate-train/test.
* **Parameter-Efficient Adaptation:** Adapters (MAD-X, BAD-X), LoRA, prefix-tuning.
* **Representation Alignment Methods:** LASER, LaBSE, InfoXLM.
* **Failure Cases:** script mismatch, typological distance, translation noise.
* **Mini-figure:** visualizing embedding overlap (t-SNE multilingual clusters).

---

### **8. Dialogue Modeling in Low-Resource Settings (~1 page)** -- Aditya

*(Focus purely on dialogue; transfer techniques already covered above.)*

* **Dialogue-Specific Fine-Tuning:** mT5-Dialog, Poly-encoder, mDialoGPT.
* **Data Augmentation for Dialogue:** back-translation, paraphrasing, bootstrapping.
* **Code-Switch & Mixed-Language Handling:** dynamic vocabularies, language tags.
* **Evaluation for Dialogue:** slot accuracy, F1, BLEU, LINCE, GLUECoS benchmarks.

---

### **9. Instruction-Tuning and the LLM Era (~1.5 pages)** -- Ruhma

* **Multilingual Instruction Models:** BLOOMZ, mT0, mT5-XXL, Aya, Yi-Intl.
* **Cross-Lingual Preference Alignment:** cultural reward models, Multi-Alpaca.
* **Emergent Capabilities:** zero-shot generalization, unseen language following.
* **Limitations:** cultural misalignment, evaluation bias, performance imbalance.
* **Trend Synthesis Figure:** pre-LLM → instruction-tuned → alignment-tuned.

---

### **10. Speech & Multimodal Extensions (~1 page)** -- Ruhma

* **Speech–Text Integration:** Whisper, SeamlessM4T, NLLB-Speech.
* **Vision–Language Expansion:** Kosmos-2, Gemini 1.5 Pro multilingual alignment.
* **Evaluation Metrics:** WER, BLEU, COMET-Kiwi, multimodal faithfulness.
* **Deployment Note:** on-device multilingual ASR/NLU efficiency.

---

### **11. Evaluation, Fairness, and Safety (~1.5 pages)** -- Isaac

*(Now clearly post-model section — behavioral analysis only.)*

* **11.1 Evaluation Benchmarks & Metrics:** XTREME, XWinograd, XSAFETY, Multi-HateCheck.

  * Metrics: BLEU, COMET, BLEURT, toxicity, calibration, value alignment.
* **11.2 Representation Fairness:** bias in embeddings (mBERT, XLM-R) across gender/religion.
* **11.3 Human Evaluation:** cross-lingual rater effects, crowdsourcing pitfalls.
* **11.4 Cultural & Safety Analysis:** hallucination, offensive generation, multilingual toxicity.

---

### **12. Open Challenges & Future Directions (~1 page)** -- Isaac

* **Truly Low-Resource Languages:** orthography, scriptless dialects.
* **Evaluation Leakage:** pretraining data contamination.
* **Cultural Grounding:** participatory data design, localized value alignment.
* **Sustainability:** compute/energy fairness.
* **Green Multilingual AI:** compression, low-compute adaptation.

---

### **13. Societal Impact, Policy, and Accessibility (~0.5 page)** -- Isaac

* **Community Initiatives:** Masakhane, BigScience, IndicNLP.
* **Policy Angle:** digital divide, UN SDGs, language preservation.
* **Equitable Access:** open-source multilingual models for education, health, and government.

---

### **14. Discussion & Synthesis (~0.5 page)**

* Connect trends: *representation → transfer → instruction → alignment*.
* Visual summary: timeline or conceptual map (2019–2025).
* Highlight emerging paradigm: *from data scarcity → cultural alignment.*

---

### **15. Conclusion (~0.5 page)**

* Reiterate inclusivity and linguistic diversity goals.
* Summarize findings: data, models, fairness, sustainability.
* End with call for participatory, lightweight, and culturally grounded multilingual AI.

---

### **16. References**

Standard ACL style (`acl_natbib.bst`).

