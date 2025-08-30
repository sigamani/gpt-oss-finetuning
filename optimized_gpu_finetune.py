#!/usr/bin/env python3
"""
Optimized GPU fine-tuning script with improved parameters and logging
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict

def setup_wandb_with_key():
    """Setup W&B with proper authentication"""
    try:
        import wandb
        
        # Check if already logged in
        if wandb.api.api_key:
            print("✅ W&B already authenticated")
        else:
            # For demo purposes, create offline run
            os.environ["WANDB_MODE"] = "offline"
            print("⚠️ W&B running in offline mode (set WANDB_API_KEY for online logging)")
        
        wandb.init(
            project="claude-mentoring-finetune-optimized",
            name="gpu-optimized-training",
            config={
                "model": "gpt2",
                "dataset_size": 40,
                "epochs": 3,
                "batch_size": 1,
                "learning_rate": 5e-5,
                "max_length": 256,
                "gradient_accumulation_steps": 16,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "attention_mask": True,
            }
        )
        return True
    except Exception as e:
        print(f"❌ W&B setup failed: {e}")
        return False

def run_optimized_gpu_finetuning():
    """Optimized GPU fine-tuning with better parameters"""
    print("🚀 Starting OPTIMIZED GPU fine-tuning...")
    
    # Setup logging
    wandb_enabled = setup_wandb_with_key()
    
    try:
        from transformers import (
            GPT2LMHeadModel,
            GPT2Tokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            EarlyStoppingCallback
        )
        from datasets import Dataset
        import torch
        
        # Check GPU
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print("📚 Loading model for GPU...")
        model_name = "gpt2"  # Use base GPT-2 for better training stability
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Add proper pad token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        
        # Resize token embeddings to account for new pad token
        model.resize_token_embeddings(len(tokenizer))
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        print("📖 Loading dataset...")
        with open("data/claude_mentoring_dataset.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        def format_conversation(messages):
            text = ""
            for msg in messages:
                if msg["role"] == "system":
                    text += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    text += f"Human: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    text += f"Assistant: {msg['content']}\n"
            text += tokenizer.eos_token
            return text
        
        texts = [format_conversation(item["messages"]) for item in data]
        dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            # Proper tokenization with attention masks
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256,  # Shorter for better training
                return_tensors=None,
                return_attention_mask=True,  # Important for quality
            )
            
            # Set labels, ignoring pad tokens in loss calculation
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            # Set pad token labels to -100 to ignore in loss
            for i in range(len(tokenized["labels"])):
                labels = tokenized["labels"][i]
                attention_mask = tokenized["attention_mask"][i]
                # Set labels to -100 where attention_mask is 0 (padding)
                tokenized["labels"][i] = [
                    label if mask == 1 else -100 
                    for label, mask in zip(labels, attention_mask)
                ]
            
            return tokenized
        
        print("🔄 Tokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        print(f"Dataset size: {len(tokenized_dataset)}")
        
        print("🎯 OPTIMIZED GPU Training setup...")
        
        # Split dataset for validation
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        training_args = TrainingArguments(
            output_dir="./claude_mentor_optimized",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,  # Effective batch size of 16
            warmup_ratio=0.1,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_steps=1,
            eval_steps=5,
            save_steps=10,
            eval_strategy="steps",
            fp16=False,
            dataloader_drop_last=False,
            report_to="wandb" if wandb_enabled else "none",
            remove_unused_columns=False,
            gradient_checkpointing=False,  # Disable for better quality
            max_grad_norm=1.0,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
        )
        
        # Custom data collator that handles attention masks properly
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        print("🚀 Starting OPTIMIZED GPU training...")
        trainer.train()
        
        print("💾 Saving OPTIMIZED GPU model...")
        trainer.save_model("./claude_mentor_optimized")
        tokenizer.save_pretrained("./claude_mentor_optimized")
        
        print("🧪 Running quick evaluation...")
        eval_results = trainer.evaluate()
        print(f"Final evaluation loss: {eval_results.get('eval_loss', 'N/A')}")
        
        if wandb_enabled:
            try:
                import wandb
                wandb.log({
                    "final_eval_loss": eval_results.get('eval_loss', 0),
                    "model_saved": True,
                    "training_completed": True
                })
                wandb.finish()
            except:
                pass
        
        print("✅ OPTIMIZED GPU fine-tuning complete!")
        
        # Create optimized test script
        test_script = '''#!/usr/bin/env python3
"""Test the optimized GPU fine-tuned model"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_optimized_model():
    print("Loading OPTIMIZED GPU fine-tuned model...")
    try:
        model = GPT2LMHeadModel.from_pretrained("./claude_mentor_optimized")
        tokenizer = GPT2Tokenizer.from_pretrained("./claude_mentor_optimized")
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
        
        test_prompts = [
            "Human: How do I create a function that checks if a number is prime?\\nAssistant:",
            "Human: Can you help me debug my Python code?\\nAssistant:",
            "Human: What's the best way to learn programming?\\nAssistant:",
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\\n{'='*60}")
            print(f"Test {i}: {prompt.split('Assistant:')[0]}")
            print(f"{'='*60}")
            
            # Proper encoding with attention mask
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=256
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("Assistant:")[-1].strip()
            print(f"Assistant: {assistant_response}")
            
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optimized_model()
'''
        
        with open("test_optimized_model.py", "w") as f:
            f.write(test_script)
        
        print("📄 Created test_optimized_model.py")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 OPTIMIZED GPU Fine-tuning with W&B Logging")
    
    if run_optimized_gpu_finetuning():
        print("🎉 SUCCESS! Optimized model in ./claude_mentor_optimized/")
        print("🧪 Test with: python3 test_optimized_model.py")
        print("📊 Check W&B dashboard for detailed training metrics")
    else:
        print("❌ Failed")