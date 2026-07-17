from .llama import llama_rec

if __name__ == '__main__':
    from transformers import AutoModel
    model_cache_dir = 'path/to/your/model_cache_dir'
    kwargs = {"cache_dir": model_cache_dir, "local_files_only": True}

    for model_name in ["bigscience/bloom-3b", "facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-6.7b", "google/gemma-2b", "meta-llama/Llama-2-7b-hf","google-bert/bert-large-cased", "THUDM/glm-2b"]:
        print('-'*100)
        print(model_name)
        model = AutoModel.from_pretrained(model_name, **kwargs)
        print(model)

   