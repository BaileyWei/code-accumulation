from pinpoint_normalization.src.processor import PostProcessor

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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

