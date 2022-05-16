[![PyPI version](https://badge.fury.io/py/text2vec-service.svg)](https://badge.fury.io/py/text2vec-service)
[![Downloads](https://pepy.tech/badge/text2vec-service)](https://pepy.tech/project/text2vec-service)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/nlp-service.svg)](https://github.com/shibing624/nlp-service/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/nlp-service.svg)](https://github.com/shibing624/nlp-service/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# nlp-service
Bert model service.

**nlp-service**搭建了一个高效的文本转向量服务。


**Guide**
- [Feature](#Feature)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Reference](#reference)


# Feature
BERT service with C/S module.

# Install
```
pip install torch # conda install pytorch
pip install -U text2vec-service
```

or

```
git clone https://github.com/shibing624/nlp-service.git
cd nlp-service
python setup.py install
```

# Usage
#### 1. Start the BERT service
After installing the server, you should be able to use `service-server-start` CLI as follows:
```bash
service-server-start -model_dir shibing624/text2vec-base-chinese -num_worker=4 
```
This will start a service with four workers, meaning that it can handle up to four **concurrent** requests. More concurrent requests will be queued in a load balancer. Details can be found in our [FAQ](#q-what-is-the-parallel-processing-model-behind-the-scene) and [the benchmark on number of clients](#speed-wrt-num_client).


<details>
 <summary>Alternatively, one can start the BERT Service in a Docker Container (click to expand...)</summary>

```bash
docker build -t nlp-service -f ./docker/Dockerfile .
NUM_WORKER=1
PATH_MODEL=/PATH_TO/_YOUR_MODEL/
docker run --runtime nvidia -dit -p 5555:5555 -p 5556:5556 -v $PATH_MODEL:/model -t nlp-service $NUM_WORKER
```
</details>


#### 2. Use Client to Get Sentence Encodes
Now you can encode sentences simply as follows:
```python
from service.client import BertClient
bc = BertClient()
bc.encode(['First do it', 'then do it right', 'then do it better'])
```
It will return a `ndarray` (or `List[List[float]]` if you wish), in which each row is a fixed-length vector representing a sentence. Having thousands of sentences? Just `encode`! *Don't even bother to batch*, the server will take care of it.



#### Use BERT Service Remotely
One may also start the service on one (GPU) machine and call it from another (CPU) machine as follows:

```python
# on another CPU machine
from service.client import BertClient
bc = BertClient(ip='xx.xx.xx.xx')  # ip address of the GPU machine
bc.encode(['First do it', 'then do it right', 'then do it better'])
```

Note that you only need `pip install -U nlp-client` in this case, the server side is not required. You may also [call the service via HTTP requests.](#using-bert-as-service-to-serve-http-requests-in-json)


<h2 align="center">Server and Client API</h2>
<p align="right"><a href="#nlp-service"><sup>▴ Back to top</sup></a></p>

[![ReadTheDoc](https://readthedocs.org/projects/bert-as-service/badge/?version=latest&style=for-the-badge)](http://bert-as-service.readthedocs.io)


### Server API

[Please always refer to the latest server-side API documented here.](https://bert-as-service.readthedocs.io/en/latest/source/server.html#server-side-api), you may get the latest usage via:
```bash
service-server-start --help
service-server-terminate --help
service-server-benchmark --help
```

| Argument | Type | Default | Description |
|--------------------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_dir` | str | *Required* | folder path of the pre-trained BERT model. |
| `max_seq_len` | int | `25` | maximum length of sequence, longer sequence will be trimmed on the right side. Set it to NONE for dynamically using the longest sequence in a (mini)batch. |
| `cased_tokenization` | bool | False | Whether tokenizer should skip the default lowercasing and accent removal. Should be used for e.g. the multilingual cased pretrained BERT model. |
| `num_worker` | int | `1` | number of (GPU/CPU) worker runs BERT model, each works in a separate process. |
| `max_batch_size` | int | `256` | maximum number of sequences handled by each worker, larger batch will be partitioned into small batches. |
| `priority_batch_size` | int | `16` | batch smaller than this size will be labeled as high priority, and jumps forward in the job queue to get result faster |
| `port` | int | `5555` | port for pushing data from client to server |
| `port_out` | int | `5556`| port for publishing results from server to client |
| `http_port` | int | None | server port for receiving HTTP requests |
| `cors` | str | `*` | setting "Access-Control-Allow-Origin" for HTTP requests |
| `gpu_memory_fraction` | float | `0.5` | the fraction of the overall amount of memory that each GPU should be allocated per worker |
| `cpu` | bool | False | run on CPU instead of GPU |
| `xla` | bool | False | enable [XLA compiler](https://www.tensorflow.org/xla/jit) for graph optimization (*experimental!*) |
| `fp16` | bool | False | use float16 precision (experimental) |
| `device_map` | list | `[]` | specify the list of GPU device ids that will be used (id starts from 0)|

### Client API

[Please always refer to the latest client-side API documented here.](https://bert-as-service.readthedocs.io/en/latest/source/client.html#module-client) Client-side provides a Python class called `BertClient`, which accepts arguments as follows:

| Argument | Type | Default | Description |
|----------------------|------|-----------|-------------------------------------------------------------------------------|
| `ip` | str | `localhost` | IP address of the server |
| `port` | int | `5555` | port for pushing data from client to server, *must be consistent with the server side config* |
| `port_out` | int | `5556`| port for publishing results from server to client, *must be consistent with the server side config* |
| `output_fmt` | str | `ndarray` | the output format of the sentence encodes, either in numpy array or python List[List[float]] (`ndarray`/`list`) |
| `show_server_config` | bool | `False` | whether to show server configs when first connected |
| `check_version` | bool | `True` | whether to force client and server to have the same version |
| `identity` | str | `None` | a UUID that identifies the client, useful in multi-casting |
| `timeout` | int | `-1` | set the timeout (milliseconds) for receive operation on the client |

A `BertClient` implements the following methods and properties:

| Method |  Description |
|--------|------|
|`.encode()`|Encode a list of strings to a list of vectors|
|`.encode_async()`|Asynchronous encode batches from a generator|
|`.fetch()`|Fetch all encoded vectors from server and return them in a generator, use it with `.encode_async()` or `.encode(blocking=False)`. Sending order is NOT preserved.|
|`.fetch_all()`|Fetch all encoded vectors from server and return them in a list, use it with `.encode_async()` or `.encode(blocking=False)`. Sending order is preserved.|
|`.close()`|Gracefully close the connection between the client and the server|
|`.status`|Get the client status in JSON format|
|`.server_status`|Get the server status in JSON format|


<h2 align="center">:book: Tutorial</h2>
<p align="right"><a href="#nlp-service"><sup>▴ Back to top</sup></a></p>

The full list of examples can be found in [`examples/`](examples). You can run each via `python example/base-demo.py`. Most of examples require you to start a BertServer first, please follow [the instruction here](#2-start-the-bert-service). 


### Building a QA semantic search engine in 3 minutes

> The complete example can [be found example8.py](example/example8.py).

As the first example, we will implement a simple QA search engine using `bert-as-service` in just three minutes. No kidding! The goal is to find similar questions to user's input and return the corresponding answer. To start, we need a list of question-answer pairs. Fortunately, this README file already contains [a list of FAQ](#speech_balloon-faq), so I will just use that to make this example perfectly self-contained. Let's first load all questions and show some statistics.

```python
prefix_q = '##### **Q:** '
with open('README.md') as fp:
    questions = [v.replace(prefix_q, '').strip() for v in fp if v.strip() and v.startswith(prefix_q)]
    print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))
```

This gives `33 questions loaded, avg. len of 9`. So looks like we have enough questions. Now start a BertServer with `uncased_L-12_H-768_A-12` pretrained BERT model:
```bash
service-server-start -num_worker=1 -model_dir=/data/cips/data/lab/data/model/uncased_L-12_H-768_A-12
```
 
Next, we need to encode our questions into vectors:
```python
bc = BertClient(port=4000, port_out=4001)
doc_vecs = bc.encode(questions)
```

Finally, we are ready to receive new query and perform a simple "fuzzy" search against the existing questions. To do that, every time a new query is coming, we encode it as a vector and compute its dot product with `doc_vecs`; sort the result descendingly; and return the top-k similar questions as follows: 
```python
while True:
    query = input('your question: ')
    query_vec = bc.encode([query])[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], questions[idx]))
```

That's it! Now run the code and type your query, see how this search engine handles fuzzy match:
<p align="center"><img src=".github/qasearch-demo.gif?raw=true"/></p>

### Serving a fine-tuned BERT model

Pretrained BERT models often show quite "okayish" performance on many tasks. However, to release the true power of BERT a fine-tuning on the downstream task (or on domain-specific data) is necessary. In this example, I will show you how to serve a fine-tuned BERT model.

We follow the instruction in ["Sentence (and sentence-pair) classification tasks"](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks) and use `run_classifier.py` to fine tune `uncased_L-12_H-768_A-12` model on MRPC task. The fine-tuned model is stored at `/tmp/mrpc_output/`, which can be changed by specifying `--output_dir` of `run_classifier.py`.

If you look into `/tmp/mrpc_output/`, it contains something like:

Don't be afraid of those mysterious files, as the only important one to us is `model.ckpt-343.data-00000-of-00001` (looks like my training stops at the 343 step. One may get `model.ckpt-123.data-00000-of-00001` or `model.ckpt-9876.data-00000-of-00001` depending on the total training steps). Now we have collected all three pieces of information that are needed for serving this fine-tuned model:
- The pretrained model is downloaded to `/path/to/bert/uncased_L-12_H-768_A-12`
- Our fine-tuned model is stored at `/tmp/mrpc_output/`;
- Our fine-tuned model checkpoint is named as `model.ckpt-343` something something.

Now start a BertServer by putting three pieces together:

```bash
service-server-start -model_dir=/tmp/mrpc_output/
```

After the server started, you should find this line in the log:
```text
I:GRAPHOPT:[gra:opt: 50]:checkpoint (override by fine-tuned model): /tmp/mrpc_output/model.ckpt-343
```
Which means the BERT parameters is overrode and successfully loaded from our fine-tuned `/tmp/mrpc_output/model.ckpt-343`. Done!

In short, find your fine-tuned model path and checkpoint name, then feed them to `-tuned_model_dir` and `-ckpt_name`, respectively.



# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/nlp-service.svg)](https://github.com/shibing624/nlp-service/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Citation

如果你在研究中使用了nlp-service，请按如下格式引用：

APA:
```latex
Xu, M. nlp-service: Bert model embedding service (Version 0.0.2) [Computer software]. https://github.com/shibing624/nlp-service
```

BibTeX:
```latex
@software{Xu_nlp-service_Text_to,
author = {Xu, Ming},
title = {{nlp-service: Bert model embedding service}},
url = {https://github.com/shibing624/nlp-service},
version = {0.0.2}
}
```

# License


授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加nlp-service的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python -m pytest -v`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

# Reference
- [jina-ai/clip-as-service](https://github.com/jina-ai/clip-as-service)
- [huggingface/transformers](https://github.com/huggingface/transformers)
