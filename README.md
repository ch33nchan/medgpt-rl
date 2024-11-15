[**🇨🇳Chinese**](https://github.com/shibing624/MedicalGPT/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/MedicalGPT/blob/main/README_EN.md) | [**📖Docs**](https://github.com/shibing624/MedicalGPT/wiki) | [**🤖Models**](https://huggingface.co/shibing624)

<div align="center">
  <a href="https://github.com/shibing624/MedicalGPT">
    <img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/logo.png" height="100" alt="Logo">
  </a>
</div>

-----------------

# MedicalGPT: Training Medical GPT Model
[![HF Models](https://img.shields.io/badge/Hugging%20Face-shibing624-green)](https://huggingface.co/shibing624)
[![Github Stars](https://img.shields.io/github/stars/shibing624/MedicalGPT?color=yellow)](https://star-history.com/#shibing624/MedicalGPT&Timeline)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/MedicalGPT.svg)](https://github.com/shibing624/MedicalGPT/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)

## 📖 Introduction

**MedicalGPT** trains a medical GPT model using the ChatGPT training pipeline, implementing Pretraining, Supervised Finetuning, RLHF (Reward Modeling and Reinforcement Learning), and DPO (Direct Preference Optimization).

**MedicalGPT** trains large medical models with incremental pretraining, supervised fine-tuning, reward-based reinforcement learning, and direct preference optimization.

<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/dpo.jpg" width="860" />

- RLHF training pipeline from Andrej Karpathy's presentation PDF [State of GPT](https://karpathy.ai/stateofgpt.pdf) and video [Video](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)
- DPO method from the paper [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)
- The ORPO method comes from the paper [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)

## 🔥 News
[2024/09/21] v2.3: Supports **[Qwen-2.5](https://qwenlm.github.io/zh/blog/qwen2.5/)** series model [Release-v2.3](https://github.com/shibing624/MedicalGPT/releases/tag/2.3.0)

[2024/08/02] v2.2: Supports role-playing model training, added a script for generating doctor-patient dialogue SFT data [role_play_data](https://github.com/shibing624/MedicalGPT/blob/main/role_play_data/README.md), see [Release-v2.2](https://github.com/shibing624/MedicalGPT/releases/tag/2.2.0)

[2024/06/11] v2.1: Supports **[Qwen-2](https://qwenlm.github.io/blog/qwen2/)** series model, see [Release-v2.1](https://github.com/shibing624/MedicalGPT/releases/tag/2.1.0)

[2024/04/24] v2.0: Supports **[Llama-3](https://huggingface.co/meta-llama)** series model, see [Release-v2.0](https://github.com/shibing624/MedicalGPT/releases/tag/2.0.0)

[2024/04/17] v1.9: Supports **[ORPO](https://arxiv.org/abs/2403.07691)**, detailed usage in `run_orpo.sh`. See [Release-v1.9](https://github.com/shibing624/MedicalGPT/releases/tag/1.9.0)

[2024/01/26] v1.8: Supports fine-tuning Mixtral MoE model **[Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)**. See [Release-v1.8](https://github.com/shibing624/MedicalGPT/releases/tag/1.8.0)

[2024/01/14] v1.7: Added retrieval-augmented generation (RAG) based file Q&A [ChatPDF](https://github.com/shibing624/ChatPDF) feature, code `chatpdf.py`, can improve industry Q&A accuracy by combining fine-tuned LLM with knowledge base file Q&A. See [Release-v1.7](https://github.com/shibing624/MedicalGPT/releases/tag/1.7.0)

[2023/10/23] v1.6: Added RoPE interpolation to extend GPT model context length; supported [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) and [LongLoRA](https://github.com/dvlab-research/LongLoRA) for LLaMA model, proposed **$S^2$-Attn**; supported [NEFTune](https://github.com/neelsjain/NEFTune) for embedding noise training method. See [Release-v1.6](https://github.com/shibing624/MedicalGPT/releases/tag/1.6.0)

## 😊 Features

Based on the ChatGPT Training Pipeline, this project implements the training of domain models—large language models in the medical industry:

- Stage 1: PT (Continue PreTraining) Incremental pretraining, secondary pretraining of the GPT model on a large amount of domain document data to adapt to the domain data distribution (optional)
- Stage 2: SFT (Supervised Fine-tuning) Supervised fine-tuning, constructing an instruction fine-tuning dataset, performing instruction fine-tuning on the pre-trained model to align instruction intentions and inject domain knowledge
- Stage 3
  - RLHF (Reinforcement Learning from Human Feedback) Reinforcement learning of language models based on human feedback, divided into two steps:
    - RM (Reward Model) Reward model modeling, constructing a human preference ranking dataset, training a reward model to model human preferences, mainly the "HHH" principle, specifically "helpful, honest, harmless"
    - RL (Reinforcement Learning) Reinforcement learning, using the reward model to train the SFT model, the generative model uses rewards or punishments to update its strategy to generate higher quality, more human-preferred text
  - [DPO (Direct Preference Optimization)](https://arxiv.org/pdf/2305.18290.pdf) Direct preference optimization method, DPO achieves precise control of language model behavior by directly optimizing the language model, without the need for complex reinforcement learning, and can effectively learn human preferences. DPO is easier to implement and train compared to RLHF, with better results
  - [ORPO](https://arxiv.org/abs/2403.07691) Optimization method without reference model, through ORPO, LLM can simultaneously learn instruction following and meet human preferences

### Release Models

| Model                                                                                                             | Base Model                                                                              | Introduction                                                                                                                                                                 |
|:------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [shibing624/ziya-llama-13b-medical-lora](https://huggingface.co/shibing624/ziya-llama-13b-medical-lora)           | [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)       | Fine-tuned a version of Ziya-LLaMA-13B model on 2.4 million bilingual medical datasets [shibing624/medical](https://huggingface.co/datasets/shibing624/medical), improved medical Q&A performance, released fine-tuned LoRA weights (single-turn dialogue)                                 |
| [shibing624/ziya-llama-13b-medical-merged](https://huggingface.co/shibing624/ziya-llama-13b-medical-merged)       | [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)       | Fine-tuned a version of Ziya-LLaMA-13B model on 2.4 million bilingual medical datasets [shibing624/medical](https://huggingface.co/datasets/shibing624/medical), improved medical Q&A performance, released fine-tuned complete model weights (single-turn dialogue)                                 |
| [shibing624/vicuna-baichuan-13b-chat-lora](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat-lora)       | [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | Fine-tuned a version of baichuan-13b-chat multi-turn Q&A model on 100,000 multilingual ShareGPT GPT4 multi-turn dialogue datasets [shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) and medical datasets [shibing624/medical](https://huggingface.co/datasets/shibing624/medical), improved daily and medical Q&A performance, released fine-tuned LoRA weights |
| [shibing624/vicuna-baichuan-13b-chat](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat)                 | [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | Fine-tuned a version of baichuan-13b-chat multi-turn Q&A model on 100,000 multilingual ShareGPT GPT4 multi-turn dialogue datasets [shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) and medical datasets [shibing624/medical](https://huggingface.co/datasets/shibing624/medical), improved daily and medical Q&A performance, released fine-tuned complete model weights |
| [shibing624/llama-3-8b-instruct-262k-chinese](https://huggingface.co/shibing624/llama-3-8b-instruct-262k-chinese) | [Llama-3-8B-Instruct-262k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k)  | Fine-tuned using ORPO method on 20,000 bilingual preference datasets [shibing624/DPO-En-Zh-20k-Preference](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference), resulting in a long-text multi-turn dialogue model suitable for RAG and multi-turn dialogue                   |

Demonstration of [shibing624/vicuna-baichuan-13b-chat](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat) model performance:
<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/demo-screen.gif" width="860" />
See specific cases in [Inference Examples](#inference-examples)

## ▶️ Demo

We provide a simple interactive web interface based on gradio. After starting the service, you can access it through a browser, input questions, and the model will return answers.

Start the service with the following command:
```shell
CUDA_VISIBLE_DEVICES=0 python gradio_demo.py --model_type base_model_type --base_model path_to_llama_hf_dir --lora_model path_to_lora_dir
```

Parameter explanation:

- `--model_type {base_model_type}`: Pre-trained model type, such as llama, bloom, chatglm, etc.
- `--base_model {base_model}`: Directory containing HF format LLaMA model weights and configuration files, or use HF Model Hub model name
- `--lora_model {lora_model}`: Directory containing LoRA files, or use HF Model Hub model name. If LoRA weights are already merged into the pre-trained model, remove the --lora_model parameter
- `--tokenizer_path {tokenizer_path}`: Directory containing the corresponding tokenizer. If not provided, its default value is the same as --base_model
- `--template_name`: Template name, such as `vicuna`, `alpaca`, etc. If not provided, its default value is vicuna
- `--only_cpu`: Use only CPU for inference
- `--resize_emb`: Whether to adjust embedding size, if not adjusted, use the pre-trained model's embedding size, default is not to adjust

## 💾 Install
#### Updating the requirements
`requirements.txt` will be updated from time to time to adapt to the latest features. Use the following command to update dependencies:

```shell
git clone https://github.com/shibing624/MedicalGPT
cd MedicalGPT
pip install -r requirements.txt --upgrade
```

#### Hardware Requirement (VRAM)

\* *Estimated values*

| Training Method  | Precision          |   7B  |  13B  |  30B  |   70B  |  110B  |  8x7B |  8x22B |
|-------|-------------| ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| Full Parameter   | AMP (Automatic Mixed Precision) | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| Full Parameter   | 16          |  60GB | 120GB | 300GB |  600GB |  900GB | 400GB | 1200GB |
| LoRA  | 16          |  16GB |  32GB |  64GB |  160GB |  240GB | 120GB |  320GB |
| QLoRA | 8           |  10GB |  20GB |  40GB |   80GB |  140GB |  60GB |  160GB |
| QLoRA | 4           |   6GB |  12GB |  24GB |   48GB |   72GB |  30GB |   96GB |
| QLoRA | 2           |   4GB |   8GB |  16GB |   24GB |   48GB |  18GB |   48GB |

## 🚀 Training Pipeline

Training Stage:

| Stage                          | Introduction | Python script                                                                                           | Shell script                                                                  |
|:-------------------------------|:-------------|:--------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
| Continue Pretraining           | Incremental Pretraining        | [pretraining.py](https://github.com/shibing624/MedicalGPT/blob/main/pretraining.py)                     | [run_pt.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_pt.sh)     |
| Supervised Fine-tuning         | Supervised Fine-tuning        | [supervised_finetuning.py](https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py) | [run_sft.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_sft.sh)   |
| Direct Preference Optimization | Direct Preference Optimization       | [dpo_training.py](https://github.com/shibing624/MedicalGPT/blob/main/dpo_training.py)                   | [run_dpo.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_dpo.sh)   |
| Reward Modeling                | Reward Modeling       | [reward_modeling.py](https://github.com/shibing624/MedicalGPT/blob/main/reward_modeling.py)             | [run_rm.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_rm.sh)     |
| Reinforcement Learning         | Reinforcement Learning         | [ppo_training.py](https://github.com/shibing624/MedicalGPT/blob/main/ppo_training.py)                   | [run_ppo.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_ppo.sh)   |
| ORPO                           | Monolithic Preference Optimization       | [orpo_training.py](https://github.com/shibing624/MedicalGPT/blob/main/orpo_training.py)                  | [run_orpo.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_orpo.sh) |

## 💻 Inference
After training is complete, we now load the trained model to verify the text generation effect of the model.

```shell
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_type base_model_type \
    --base_model path_to_model_hf_dir \
    --tokenizer_path path_to_model_hf_dir \
    --lora_model path_to_lora \
    --interactive
```

Parameter explanation:

- `--model_type {base_model_type}`: Pre-trained model type, such as llama, bloom, chatglm, etc.
- `--base_model {base_model}`: Directory containing HF format LLaMA model weights and configuration files
- `--tokenizer_path {base_model}`: Directory containing HF format LLaMA model weights and configuration files
- `--lora_model {lora_model}`: Directory containing LoRA files, or use HF Model Hub model name. If LoRA weights are already merged into the pre-trained model, you can omit this parameter
- `--tokenizer_path {tokenizer_path}`: Directory containing the corresponding tokenizer. If not provided, its default value is the same as --base_model
- `--template_name`: Template name, such as `vicuna`, `alpaca`, etc. If not provided, its default value is vicuna
- `--interactive`: Start multi-turn Q&A in interactive mode, using streaming inference
- `--data_file {file_name}`: In non-interactive mode, read the content in file_name for batch prediction
- `--output_file {file_name}`: In non-interactive mode, write the predicted results in jsonl format to file_name
- `--resize_emb`: Whether to adjust embedding size, if not adjusted, use the pre-trained model's embedding size, default is not to adjust
- `--only_cpu`: Use only CPU for inference
- `--gpus {gpu_ids}`: Specify the GPU device number to use, default is 0. If using multiple GPUs, separate with commas, such as 0,1,2

#### Multi-GPU Inference
Multi-GPU data parallelism, batch inference
```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 inference_multigpu_demo.py --model_type baichuan --base_model shibing624/vicuna-baichuan-13b-chat
```
#### Vllm Multi-GPU Deployment
```shell
bash run_Multi-GPUs-deployment.sh
```

#### Inference Examples
[shibing624/vicuna-baichuan-13b-chat](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat) inference examples:

<details><summary>Model output case</summary>

| Human                                                                              | Assistant                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| What to do if a child has a fever?                                                                            | Fever is a natural response of the body to infection or other diseases. Here are some suggestions to help alleviate fever symptoms:<br/>1. Keep the body hydrated: During a fever, the body loses water, so it is necessary to stay hydrated.<br/>2. Use an ice pack: Placing an ice pack on the forehead or neck can help lower body temperature.<br/>3. Use antipyretics: Antipyretics can help lower body temperature.<br/>4. Maintain body temperature: Maintaining body temperature can help lower body temperature.<br/>5. Use cold or warm water to wipe the body: Using cold or warm water to wipe the body can help lower body temperature.<br/>6. Use antipyretics: Antipyretics can help lower body temperature.<br/>7. Maintain body temperature: Maintaining body temperature can help lower body temperature.<br/>8. Use cold or warm water to wipe the body: Using cold or warm water to wipe the body can help lower body temperature.<br/>9. Use antipyretics: Antipyretics can help lower body temperature.<br/>10. Maintain body temperature: Maintaining body temperature can help lower body temperature.                                                                                                                                                                                                                                                         |
| What medicine can a one-year-old baby take for a fever?                                                                        | When a one-year-old baby has a fever, you can consider the following medications:<br/>1. Antipyretics: Such as Ibuprofen or Paracetamol. These medications can help lower body temperature and relieve fever symptoms.<br/>2. Antibiotics: If the fever is caused by a bacterial infection, antibiotics can be considered. But be sure to use them under the advice of a doctor.<br/>3. Antiviral drugs: If the fever is caused by a viral infection, antiviral drugs can be considered. But be sure to use them under the advice of a doctor.<br/>4. Antipyretic and analgesic drugs: Such as Aspirin or Paracetamol. These medications can help relieve fever symptoms.<br/>5. Antihistamines: Such as Diphenhydramine or Naproxen. These medications can help relieve fever symptoms.<br/>Be sure to use these medications under the advice of a doctor and follow the dosage and usage instructions on the medication label. At the same time, keep the baby hydrated and ensure the baby gets enough rest.                                                                                                                                                                                                                             |
| What diseases can anal lesions be a symptom of?                                                                     | Anal lesions can be caused by a variety of reasons, including infections, inflammation, tumors, or other diseases. Some common anal lesions include:<br/>1. Anal fissure: This is a tear in the skin of the anus, usually caused by constipation or other pressure factors.<br/>2. Anal fistula: This is an anal infection that leads to the formation of a fistula between the skin of the anus and the rectum.<br/>3. Anal warts: These are wart-like lesions on the skin of the anus, usually caused by the human papillomavirus (HPV).<br/>4. Anal abscess: This is an anal infection that leads to the formation of an abscess under the skin of the anus.<br/>5. Anal tumor: This is a tumor that forms on the skin of the anus or inside the rectum.<br/>6. Anal ulcer: This is an ulcer that forms on the skin of the anus, usually caused by infection or inflammation.<br/>7. Anal itching: This is a symptom of itching in the skin of the anus, which may be caused by infection, inflammation, or other diseases.<br/>If you have symptoms of anal lesions, please consult a healthcare professional for proper diagnosis and treatment.                                                                                                                                                                                                                                                                  |
| Doctor, why is there an increase in yellowish discharge?                                                                   | Hello, an increase in yellowish discharge can be caused by a variety of reasons, including infections, hormonal changes, vaginal dryness, or other gynecological issues. Here are some possible causes:<br/>1. Infection: An increase in yellowish discharge can be caused by a vaginal or urinary tract infection. Infections can be caused by bacteria, viruses, or fungi.<br/>2. Hormonal changes: Hormonal changes, such as during the menstrual cycle or menopause, can lead to an increase in yellowish discharge.<br/>3. Vaginal dryness: Vaginal dryness can lead to an increase in yellowish discharge.<br/>4. Other gynecological issues: Other gynecological issues, such as cervicitis, endometritis, or endometriosis, can also lead to an increase in yellowish discharge.<br/>If you are concerned about your discharge, it is recommended to consult a gynecologist for examination and diagnosis. The doctor can determine the cause based on your symptoms and examination results and provide appropriate treatment.                                                                                                                                                                                                                                                                                                                          |
| Is chocolate-colored discharge after a medical abortion a sign of incomplete abortion?                                                                    | Chocolate-colored discharge after a medical abortion does not necessarily mean that the abortion is incomplete. Chocolate-colored discharge is caused by the shedding of the uterine lining and bleeding, which usually occurs for a period of time after a medical abortion.<br/><br/>The reasons for chocolate-colored discharge after a medical abortion may include:<br/><br/>1. Shedding of the uterine lining: After a medical abortion, the uterine lining will shed, leading to bleeding. This bleeding may manifest as chocolate-colored discharge.<br/>2. Infection: After a medical abortion, there may be trauma to the uterine lining and cervix, making them susceptible to bacterial infection. Infection may lead to chocolate-colored discharge.<br/>3. Hormonal changes: After a medical abortion, hormonal levels may change, leading to chocolate-colored discharge.<br/><br/>If chocolate-colored discharge persists for a long time, or is accompanied by other symptoms such as abdominal pain, fever, or foul-smelling discharge, it is recommended to seek medical attention promptly for relevant examinations, such as a gynecological examination or ultrasound, to determine if there is an infection or other issues. Under the guidance of a doctor, appropriate treatment can be carried out.                                                                                                                                                                                                                                                                                      |
| What diseases can tretinoin and vitamin E cream treat?                                                                     | Tretinoin and vitamin E cream is a topical medication mainly used to treat acne, seborrheic dermatitis, and other skin diseases. It can help reduce sebum secretion, relieve inflammation, and promote skin repair.<br/><br/>It should be noted that tretinoin and vitamin E cream is not a panacea and may not be suitable for certain skin diseases. Before use, it is recommended to consult a doctor or pharmacist to understand the indications, contraindications, dosage, and usage of the medication. At the same time, follow the doctor's advice when using the medication and avoid self-misuse.                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| What causes acne?                                                                        | Acne, also known as pimples, is caused by clogged skin pores. There are many reasons for clogged pores, including:<br/>1. Overactive sebaceous glands: The sebaceous glands produce too much oil, leading to clogged pores.<br/>2. Clogged pores: Dead skin cells, oil, and bacteria on the skin surface clog the pores.<br/>3. Inflammation: Acne is often accompanied by inflammation, which may be caused by skin infection or the immune system's response to acne.<br/>4. Genetic factors: Some people may be more prone to acne, which may be due to genetic factors.<br/>5. Hormonal changes: Acne is more common during puberty and menstruation, which may be due to hormonal changes.<br/>6. Lifestyle factors: Certain lifestyle factors, such as stress, poor diet, and lack of exercise, may also lead to acne.<br/>7. Cosmetics and skincare products: Certain cosmetics and skincare products may cause acne, especially those containing irritating ingredients.<br/>8. Medications: Certain medications, such as oral contraceptives, may also cause acne.<br/>To prevent acne, you can take some measures, such as keeping the skin clean, using gentle skincare products, avoiding excessive use of cosmetics and skincare products, maintaining a healthy lifestyle and diet, avoiding stress and hormonal changes. If acne is severe, you may need to seek help from a dermatologist.                                                                                                                                                         |
| What should I do if I have acne on my face, what medicine can I take, and what should I pay attention to?                                                               | Acne, also known as pimples, is a common skin disease. The treatment of acne requires comprehensive consideration of diet, lifestyle, medication, and other aspects. Here are some suggestions:<br/><br/>1. Diet:<br/>* Avoid high-sugar, high-fat, and high-oil foods, such as fast food, desserts, and fried foods.<br/>* Eat more fresh vegetables, fruits, and whole grains to increase fiber intake and promote intestinal peristalsis.<br/>* Avoid excessive intake of caffeine, alcohol, spicy foods, and other stimulating foods.<br/><br/>2. Lifestyle:<br/>* Maintain a good sleep schedule and ensure adequate sleep.<br/>* Avoid prolonged use of computers, mobile phones, and other electronic products to prevent radiation from irritating the skin.<br/>* Keep the skin clean and use gentle cleansing products to clean the skin every morning and evening.<br/>* Avoid squeezing acne to prevent infection and scarring.<br/><br/>3. Medication:<br/>* Topical medications: You can use skincare products containing salicylic acid, benzoyl peroxide, azelaic acid, and other ingredients to help remove dead skin cells on the skin surface and reduce the formation of acne.<br/>* Oral medications: If acne is severe, you can consider oral antibiotics, vitamin B complex, zinc, and other medications, but they should be used under the guidance of a doctor.<br/><br/>4. Attention:<br/>* Do not use cosmetics and skincare products indiscriminately to avoid irritating the skin.<br/>* Do not over-cleanse the skin to avoid damaging the skin barrier.<br/>* Maintain a good mood and avoid negative emotions such as anxiety and depression.<br/><br/>In short, the treatment of acne requires comprehensive consideration of diet, lifestyle, medication, and other aspects. During the treatment process, you should maintain patience and confidence, follow the doctor's advice, and gradually improve the skin condition. |
| Can vas deferens obstruction cause male infertility? Doctor, hello, my colleague has been unhappy these days, I asked her what was wrong, and she said her husband has a vas deferens obstruction, so they haven't had children, it's not that they don't want to, can vas deferens obstruction cause male infertility? | Vas deferens obstruction may cause male infertility. The vas deferens is a tube that connects the testicles to the urethra and is responsible for transporting sperm from the testicles to the urethra. If the vas deferens is obstructed, sperm cannot pass through the vas deferens to reach the urethra, leading to infertility.<br/><br/>There are many causes of vas deferens obstruction, including congenital developmental abnormalities, inflammation, tumors, trauma, etc. The treatment methods for vas deferens obstruction include surgical unblocking, medication, etc. If the vas deferens obstruction is severe, surgical unblocking may be required.<br/><br/>It should be noted that vas deferens obstruction does not necessarily lead to infertility because there are other ways to transport sperm to the female body. For example, sperm can reach the female body through the epididymis, prostate, and other tissues. Therefore, if there is a vas deferens obstruction, it is recommended to seek medical attention promptly and seek help from a professional doctor.                                                                                                                                                                                                                                                                                                                                                     |
| Can myocardial infarction be treated?                                                                          | Myocardial infarction is a serious disease, and if treated promptly, it can save lives and prevent long-term disability. The key to treating myocardial infarction is early recognition and treatment.<br/><br/>The treatment of myocardial infarction usually includes the following steps:<br/><br/>1. Seek medical attention immediately: If you experience symptoms such as chest pain, shortness of breath, or fainting, seek medical attention immediately.<br/><br/>2. Medication: The doctor may prescribe medications such as nitroglycerin or aspirin to help relieve symptoms and prevent further damage.<br/><br/>3. Thrombolytic therapy: If the patient experiences a myocardial infarction before arriving at the hospital, the doctor may perform thrombolytic therapy by injecting thrombolytic drugs to help restore blood flow.<br/><br/>4. Surgical treatment: If medication or thrombolytic therapy is ineffective, the doctor may recommend surgery, such as coronary artery bypass surgery or heart stent implantation.<br/><br/>5. Rehabilitation and prevention: The treatment of myocardial infarction usually includes a rehabilitation plan to help the patient restore physical function and prevent future heart events. In addition, patients should follow the doctor's advice to reduce the risk of heart disease.<br/><br/>It should be noted that the treatment of myocardial infarction depends on the patient's specific condition and symptoms. Therefore, if you or someone you know experiences symptoms of myocardial infarction, seek medical attention immediately.                                                                                                                                                            |
| How to stay energetic, give 5 suggestions                                                                    | 1. Get enough sleep: Sleep is crucial for staying energetic. Make sure to get 7-9 hours of high-quality sleep every night.<br/>2. Balanced diet: A balanced diet is crucial for maintaining energy levels. Make sure to consume enough protein, carbohydrates, and healthy fats.<br/>3. Regular exercise: Regular exercise can help improve energy levels. Try to engage in at least 30 minutes of aerobic exercise every day, such as walking, running, or cycling.<br/>4. Manage stress: Stress can deplete energy, so managing stress is crucial for staying energetic. Try to find ways to relax and reduce stress, such as meditation, yoga, or deep breathing exercises.<br/>5. Stay hydrated: Dehydration can lower energy levels. Make sure to drink enough water every day and replenish fluids as needed.                                                                                                                                                                                                                                                                                                                                                         |
| Write a poem about spring                                                                             | Spring,<br/>is the season when the earth awakens,<br/>is the season when everything revives,<br/>is the season when life blooms.<br/>Spring,<br/>is the season of bright sunshine,<br/>is the season of birds singing and flowers blooming,<br/>is the season of lush greenery,<br/>is the season of infinite hope.<br/>Spring,<br/>is the season of people's laughter,<br/>is the season of children's play,<br/>is the season of lovers' embrace,<br/>is the season of people pursuing their dreams.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Why are there no polar bears in Antarctica?                                                                        | There are no polar bears in Antarctica because polar bears live in the Arctic region, while Antarctica is located in the Southern Hemisphere, with a climate and environment very different from the Arctic region. Antarctica's climate is extremely cold, covered with ice and snow all year round, while polar bears live in the Arctic region, where there is an abundance of marine life and terrestrial mammals. Additionally, Antarctica's ecosystem is different from the Arctic region, lacking the conditions necessary for polar bears to survive.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

</details>

## 📚 Dataset
### Medical Datasets

- 2.4 million Chinese medical datasets (including pretraining, instruction fine-tuning, and reward datasets): [shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- 220,000 Chinese medical dialogue datasets (Hua Tuo project): [shibing624/huatuo_medical_qa_sharegpt](https://huggingface.co/datasets/shibing624/huatuo_medical_qa_sharegpt) [Supported format]

### General Datasets

#### Pretraining datasets
- 16GB bilingual unsupervised, parallel corpus [Linly-AI/Chinese-pretraining-dataset](https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset)
- 524MB Chinese Wikipedia corpus [wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)

#### Supervised fine-tuning datasets
- 100,000 multilingual ShareGPT GPT4 multi-turn dialogue datasets: [shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) [Supported format]
- 90,000 English ShareGPT multi-turn dialogue datasets: [anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) [Supported format]
- 500,000 Chinese ChatGPT instruction Belle datasets: [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- 1 million Chinese ChatGPT instruction Belle datasets: [BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- 50,000 English ChatGPT instruction Alpaca datasets: [50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)
- 20,000 Chinese ChatGPT instruction Alpaca datasets: [shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
- 690,000 Chinese instruction Guanaco datasets (Belle 500,000 + Guanaco 190,000): [Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)
- 50,000 English ChatGPT multi-turn dialogue datasets: [RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)
- 800,000 Chinese ChatGPT multi-turn dialogue datasets: [BelleGroup/multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- 1.16 million Chinese ChatGPT multi-turn dialogue datasets: [fnlp/moss-002-sft-data](https://huggingface.co/datasets/fnlp/moss-002-sft-data)
- 38,000 Chinese ShareGPT multi-turn dialogue datasets: [FreedomIntelligence/ShareGPT-CN](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-CN)
- 1.3 million Chinese fine-tuning datasets (summary): [zhuangxialie/Llama3-Chinese-Dataset](https://modelscope.cn/datasets/zhuangxialie/Llama3-Chinese-Dataset/dataPeview) [Supported format]
- 7,000 Chinese role-playing multi-turn dialogue datasets: [shibing624/roleplay-zh-sharegpt-gpt4-data](https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) [Supported format]

#### Preference datasets
- 20,000 bilingual preference datasets: [shibing624/DPO-En-Zh-20k-Preference](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference) [Supported format]
- Original oasst1 dataset: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- 20,000 multilingual oasst1 reward datasets: [tasksource/oasst1_pairwise_rlhf_reward](https://huggingface.co/datasets/tasksource/oasst1_pairwise_rlhf_reward)
- 110,000 English hh-rlhf reward datasets: [Dahoas/full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf)
- 90,000 English reward datasets (from Anthropic's Helpful Harmless dataset): [Dahoas/static-hh](https://huggingface.co/datasets/Dahoas/static-hh)
- 70,000 English reward datasets (same source as above): [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)
- 70,000 Traditional Chinese reward datasets (translated from rm-static) [liswei/rm-static-m2m100-zh](https://huggingface.co/datasets/liswei/rm-static-m2m100-zh)
- 70,000 English Reward datasets: [yitingxie/rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets)
- 3,000 Chinese Zhihu Q&A preference datasets: [liyucheng/zhihu_rlhf_3k](https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k)

## ☎️ Contact

- Issue (Suggestions)
  : [![GitHub issues](https://img.shields.io/github/issues/shibing624/MedicalGPT.svg)](https://github.com/shibing624/MedicalGPT/issues)
- Email me: xuming: xuming624@qq.com
- WeChat me: Add my *WeChat ID: xuming624, Note: Name-Company-NLP* to join the NLP group chat (add me to join the group).

<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/wechat.jpeg" width="200" />

<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/wechat_group.jpg" width="200" />

## ⚠️ LICENSE

This project is for research purposes only. The project developers are not responsible for any harm or loss caused by using this project (including but not limited to data, models, code, etc.). Please refer to the [Disclaimer](https://github.com/shibing624/MedicalGPT/blob/main/DISCLAIMER) for details.

The code of the MedicalGPT project is licensed under [The Apache License 2.0](/LICENSE), and the code can be used for commercial purposes for free, but the model weights and data can only be used for research purposes. Please include the link and license of MedicalGPT in the product description.

## 😇 Citation

If you use MedicalGPT in your research, please cite it as follows:

```latex
@misc{MedicalGPT,
  title={MedicalGPT: Training Medical GPT Model},
  author={Ming Xu},
  year={2023},
  howpublished={\url{https://github.com/shibing624/MedicalGPT}},
}
```

## 😍 Contribute

The project code is still rough. If you have any improvements to the code, please submit them back to this project. Before submitting, note the following two points:

- Add corresponding unit tests in `tests`
- Use `python -m pytest` to run all unit tests and ensure all tests pass

After that, you can submit a PR.

## 💕 Acknowledgements

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/finetune.py)
- [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [dvlab-research/LongLoRA](https://github.com/dvlab-research/LongLoRA)

Thanks for their great work!

#### Related Projects
- [shibing624/ChatPilot](https://github.com/shibing624/ChatPilot): Provides a simple and easy-to-use Web UI interface for LLM Agent (including RAG, online search, Code interpreter)