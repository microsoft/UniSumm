import json
import os
import sys
import time
import uuid

import requests
from tqdm import tqdm


def setup_translation_keys(translator_subscription_key=None,
                           translator_endpoint=None):
    """
    Add the translation keys either from the OS environment
    or the command line args.
    Need to have a subscription key and target endpoint.
    """
    #
    key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
    if key_var_name not in os.environ and translator_subscription_key is None:
        raise Exception(
            "Please set {} in the os environment or "
            "pass as an argument.".format(key_var_name))
    elif translator_subscription_key is None and key_var_name in os.environ:
        translator_subscription_key = os.environ[key_var_name]
    endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
    if endpoint_var_name not in os.environ and translator_endpoint is None:
        raise Exception(
            "Please set {} in the os environment or pass as"
            "an argument.".format(
                endpoint_var_name))
    elif translator_endpoint is None and endpoint_var_name in os.environ:
        translator_endpoint = os.environ[key_var_name]
    return translator_subscription_key, translator_endpoint


def translate(txt_lst, targets, translation_key=None, translation_endpoint=None,
              src_lang=None, include_alignment=True, use_html=False):
    if not isinstance(targets, list):
        targets = [targets]
    assert targets
    if translation_key is None or translation_endpoint is None \
            or not bool(translation_key) or not bool(translation_endpoint):
        raise Exception(
            'Please include the necessary translation key and endpoint.'
            'Got: {} and {}'.format(
                translation_key, translation_endpoint))
    subscription_key = translation_key

    endpoint = translation_endpoint
    path = 'translate?api-version=3.0&'
    params = 'to={}'.format(targets[0])
    if len(targets) > 1:
        params += ''.join(['&to={}'.format(lang) for lang in targets[1:]])
    if src_lang is not None:
        params += '&from=%s' % src_lang
    constructed_url = endpoint + path + params
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': 'southcentralus',
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [
        {'text': t} for t in txt_lst
    ]
    request = requests.post(constructed_url, headers=headers, json=body)
    if request.status_code == 200:
        return request.json()
    if request.json()['error']['code'] == 429001:
        raise ConnectionError(
            "Requests throttled. Need to wait. Body: {}".format(
                request.json()))
    raise Exception(
        "Request failed. Request response: {} Body: {}".format(
            request.json(), body))


def try_translate_backoff(claim, targets, num_tries=8,
                          translation_key=None,
                          translation_endpoint=None,
                          src_lang=None, include_alignment=True,
                          use_html=False):
    for i in range(num_tries):
        try:
            translated_data = translate(
                claim, targets,
                translation_key=translation_key,
                translation_endpoint=translation_endpoint,
                src_lang=src_lang,
                include_alignment=include_alignment,
                use_html=use_html)
            break
        except TypeError:
            raise
        except ConnectionError as ex:
            if i == num_tries - 1:
                print("Tried {} times to translate. Raising {}.".format(i, ex))
                raise
            time_to_sleep = 2 ** (i + 2)  # wait up to a minute
            time.sleep(time_to_sleep)
    return translated_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Text to translate")
    parser.add_argument(
        "-e", "--endpoint", type=str,
        required=False,
        default="https://api.cognitive.microsofttranslator.com/",
        help="Cognitive Services Translator endpoint.")
    parser.add_argument("-k", "--key", type=str,
                        help="Cognitive Services key")
    parser.add_argument("-t", "--targets", type=str,
                        default=['zh-cn'],
                        required=False,
                        nargs='+', help="Target languages")
    parser.add_argument('--use-html', default=False, action='store_true',
                        help='include if text includes HTML/XML tags')

    args = parser.parse_args()
    key, endpoint = setup_translation_keys(args.key, args.endpoint)

    fp = 'F:/Projects/Mainz/data/july/segment.InternalTownhall.train.json'
    save_fp = 'F:/Projects/Mainz/data/july/segment.InternalTownhall.paraphrased.de1.train.json'
    jobj = json.load(open(fp, encoding='utf-8'))

    meetings = jobj['Input']['DataSets'][0]['Meetings'][:30]

    with open(save_fp, 'w', encoding='utf-8') as save:
        for m_id, m in tqdm(enumerate(meetings)):
            transcripts = m['Transcript']['Transcripts']
            utts = [u['AutoRecognizedText'] for u in transcripts]

            ulen = len(utts)
            paraphrased_utts = []
            for i in range(0, ulen, 30):
                es_result = translate(utts[i:i + 30], ['es'], key, endpoint,
                                      use_html=args.use_html, src_lang='de')

                es_utts = [u['translations'][0]['text'] for u in es_result]

                time.sleep(2)

                paraphrased = translate(es_utts, ['en'], key, endpoint,
                                        use_html=args.use_html, src_lang='de', )

                paraphrased_utts.extend([u['translations'][0]['text'] for u in paraphrased])

            assert len(utts) == len(paraphrased_utts)
            for i in range(ulen):
                m['Transcript']['Transcripts'][i]['ParaphrasedText'] = paraphrased_utts[i]
            time.sleep(5)
        json.dump(jobj, save)
