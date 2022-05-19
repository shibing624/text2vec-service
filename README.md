[![PyPI version](https://badge.fury.io/py/text2vec-service.svg)](https://badge.fury.io/py/text2vec-service)
[![Downloads](https://pepy.tech/badge/text2vec-service)](https://pepy.tech/project/text2vec-service)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/text2vec-service.svg)](https://github.com/shibing624/text2vec-service/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.7%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/text2vec-service.svg)](https://github.com/shibing624/text2vec-service/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# text2vec-service
Bert model to vector service.

**text2vec-service**搭建了一个高效的文本转向量(Text-To-Vector)服务。


**Guide**
- [Feature](#Feature)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Reference](#reference)


# Feature
BERT service with C/S.

# Install
```
pip install torch # conda install pytorch
pip install -U text2vec-service
```

or

```
git clone https://github.com/shibing624/text2vec-service.git
cd text2vec-service
python setup.py install
```

# Usage
#### 1. Start the BERT service
After installing the server, you should be able to use `service-server-start` CLI as follows:
```bash
service-server-start -model_dir shibing624/text2vec-base-chinese 
```
This will start a service with four workers, meaning that it can handle up to four **concurrent** requests. 
More concurrent requests will be queued in a load balancer. 


<details>
 <summary>Alternatively, one can start the BERT Service in a Docker Container (click to expand...)</summary>

```bash
docker build -t text2vec-service -f ./docker/Dockerfile .
NUM_WORKER=1
PATH_MODEL=/PATH_TO/_YOUR_MODEL/
docker run --runtime nvidia -dit -p 5555:5555 -p 5556:5556 -v $PATH_MODEL:/model -t text2vec-service $NUM_WORKER
```
</details>


#### 2. Use Client to Get Sentence Encodes
Now you can encode sentences simply as follows:
```python
from service.client import BertClient
bc = BertClient()
bc.encode(['如何更换花呗绑定银行卡', '花呗更改绑定银行卡'])
```
It will return a `ndarray` (or `List[List[float]]` if you wish), in which each row is a fixed-length vector 
representing a sentence. Having thousands of sentences? Just `encode`! *Don't even bother to batch*, 
the server will take care of it.



#### Use BERT Service Remotely
One may also start the service on one (GPU) machine and call it from another (CPU) machine as follows:

```python
# on another CPU machine
from service.client import BertClient
bc = BertClient(ip='xx.xx.xx.xx')  # ip address of the GPU machine
bc.encode(['如何更换花呗绑定银行卡', '花呗更改绑定银行卡'])
```


<h2 align="center">Server and Client API</h2>
<p align="right"><a href="#text2vec-service"><sup>▴ Back to top</sup></a></p>


### Server API

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
<p align="right"><a href="#text2vec-service"><sup>▴ Back to top</sup></a></p>

The full list of examples can be found in [`examples/`](examples). You can run each via `python examples/base-demo.py`.


### Serving a fine-tuned BERT model

Pretrained BERT models often show quite "okayish" performance on many tasks. However, to release the true power of 
BERT a fine-tuning on the downstream task (or on domain-specific data) is necessary. 

In this example, serve a fine-tuned BERT model.

```bash
service-server-start -model_dir shibing624/bert-base-chinese
```


### Asynchronous encoding

> The complete example can be found [examples/async_demo.py](examples/async_demo.py).

`BertClient.encode()` offers a nice synchronous way to get sentence encodes. 
However, sometimes we want to do it in an asynchronous manner by feeding all textual data to the server first, 
fetching the encoded results later. This can be easily done by:
```python
# an endless data stream, generating data in an extremely fast speed
def text_gen():
    while True:
        yield lst_str  # yield a batch of text lines

bc = BertClient()

# get encoded vectors
for j in bc.encode_async(text_gen(), max_num_batch=10):
    print('received %d x %d' % (j.shape[0], j.shape[1]))
```

### Broadcasting to multiple clients

> example: [examples/multicast_demo.py](examples/multicast_demo.py).

The encoded result is routed to the client according to its identity. If you have multiple clients with 
same identity, then they all receive the results! You can use this *multicast* feature to do some cool things, 
e.g. training multiple different models (some using `scikit-learn` some using `pytorch`) in multiple 
separated processes while only call `BertServer` once. In the example below, `bc` and its two clones will 
all receive encoded vector.

```python
# clone a client by reusing the identity 
def client_clone(id, idx):
    bc = BertClient(identity=id)
    for j in bc.listen():
        print('clone-client-%d: received %d x %d' % (idx, j.shape[0], j.shape[1]))

bc = BertClient()
# start two cloned clients sharing the same identity as bc
for j in range(2):
    threading.Thread(target=client_clone, args=(bc.identity, j)).start()

for _ in range(3):
    bc.encode(lst_str)
```

### Monitoring the service status in a dashboard

> The complete example can [be found in plugin/dashboard/](plugin/dashboard).

As a part of the infrastructure, one may also want to monitor the service status and show it in a dashboard. To do that, we can use:
```python
bc = BertClient(ip='server_ip')

json.dumps(bc.server_status, ensure_ascii=False)
```

This gives the current status of the server including number of requests, number of clients etc. in JSON format. The only thing remained is to start a HTTP server for returning this JSON to the frontend that renders it.

Alternatively, one may simply expose an HTTP port when starting a server via:

```bash
bert-serving-start -http_port 8081
```

This will allow one to use javascript or `curl` to fetch the server status at port 8081.

`plugin/dashboard/index.html` shows a simple dashboard based on Bootstrap and Vue.js.

<p align="center"><img src="docs/dashboard.png?raw=true"/></p>

### Using `text2vec-service` to serve HTTP requests in JSON

Besides calling `text2vec-service` from Python, one can also call it via HTTP request in JSON. It is quite 
useful especially when low transport layer is prohibited. Behind the scene, `text2vec-service` spawns a Flask 
server in a separate process and then reuse a `BertClient` instance as a proxy to communicate with the ventilator.

To enable the build-in HTTP server, we need to first (re)install the server with some extra Python dependencies:
```bash
pip install -U text2vec-service[http]
```

Then simply start the server with:
```bash
bert-serving-start -model_dir=/YOUR_MODEL -http_port 8081
```

Done! Your server is now listening HTTP and TCP requests at port `8081` simultaneously!

To send a HTTP request, first prepare the payload in JSON as following:
```json
{
    "id": 123,
    "texts": ["hello world", "good day!"]
}
```
, where `id` is a unique identifier helping you to synchronize the results.

Then simply call the server at `/encode` via HTTP POST request. You can use javascript or whatever, here is an 
example using `curl`:
```bash
curl -X POST http://xx.xx.xx.xx:8081/encode \
  -H 'content-type: application/json' \
  -d '{"id": 123,"texts": ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']}'
```
, which returns a JSON:
```json
{
    "id": 123,
    "results": [[768 float-list], [768 float-list]],
    "status": 200
}
```

To get the server's status and client's status, you can send GET requests at `/status/server` and `/status/client`, 
respectively.

Finally, one may also config CORS to restrict the public access of the server by specifying `-cors` when 
starting `bert-serving-start`. By default `-cors=*`, meaning the server is public accessible.


### Starting `BertServer` from Python

Besides shell, one can also start a `BertServer` from python. Simply do
```python
from service.server.helper import get_args_parser
from service.server import BertServer
args = get_args_parser().parse_args(['-model_dir', 'YOUR_MODEL_PATH_HERE',
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     '-max_seq_len', 'NONE',
                                     '-mask_cls_sep',
                                     '-cpu'])
server = BertServer(args)
server.start()
``` 

Note that it's basically mirroring the arg-parsing behavior in CLI, so everything in that `.parse_args([])` list 
should be string, e.g. `['-port', '5555']` not `['-port', 5555]`.

To shutdown the server, you may call the static method in `BertServer` class via with args:
```python
shut_args = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
BertServer.shutdown(shutdown_args)
```

Or via shell CLI:
```bash
bert-serving-terminate -port 5555
```

This will terminate the server running on localhost at port 5555. You may also use it to terminate a remote server, 
see `bert-serving-terminate --help` for details.



# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/text2vec-service.svg)](https://github.com/shibing624/text2vec-service/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Citation

如果你在研究中使用了text2vec-service，请按如下格式引用：

APA:
```latex
Xu, M. text2vec-service: Bert model embedding service (Version 0.0.2) [Computer software]. https://github.com/shibing624/text2vec-service
```

BibTeX:
```latex
@software{Xu_text2vec-service_Text_to,
author = {Xu, Ming},
title = {{text2vec-service: Bert model embedding service}},
url = {https://github.com/shibing624/text2vec-service},
version = {0.0.2}
}
```

# License


授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加text2vec-service的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python -m pytest -v`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

# Reference
- [jina-ai/clip-as-service](https://github.com/jina-ai/clip-as-service)
- [huggingface/transformers](https://github.com/huggingface/transformers)
