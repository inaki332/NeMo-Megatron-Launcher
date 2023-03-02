from argparse import ArgumentParser
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from transformers import GPT2Tokenizer

def fill_input(name, data):
    infer_input = httpclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    infer_input.set_data_from_numpy(data)
    return infer_input
def build_request(query, host, output):
    with httpclient.InferenceServerClient(host) as client:
        request_data = []
        request = np.array([query]).astype(np.uint32)
        request_len = np.array([[len(query)]]).astype(np.uint32)
        request_output_len = np.array([[output]]).astype(np.uint32)
        top_k = np.array([[1]]).astype(np.uint32)
        top_p = np.array([[0.0]]).astype(np.float32)
        temperature = np.array([[1.0]]).astype(np.float32)
        request_data.append(fill_input('input_ids', request))
        request_data.append(fill_input('input_lengths', request_len))
        request_data.append(fill_input('request_output_len', request_output_len))
        request_data.append(fill_input('runtime_top_k', top_k))
        request_data.append(fill_input('runtime_top_p', top_p))
        request_data.append(fill_input('temperature', temperature))
        result = client.infer('gpt3_5b', request_data)
        output = result.as_numpy('output_ids').squeeze()
        return output
def main():
    parser = ArgumentParser('Simple Triton Inference Requestor')
    parser.add_argument('query', type=str, help='Enter a text query to send to '
                        'the Triton Inference Server in quotes.')
    parser.add_argument('--output-length', type=int, help='Specify the desired '
                        'length for output.', default=30)
    parser.add_argument('--server', type=str, help='Specify the host:port that '
                        'Triton is listening on. Defaults to localhost:8000',
                        default='localhost:8000')
    args = parser.parse_args()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    query = tokenizer(args.query).input_ids
    request = build_request(query, args.server, args.output_length)
    print(tokenizer.decode(request))
if __name__ == '__main__':
    main()
