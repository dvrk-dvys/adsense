from apify_client import ApifyClient
import json
import html_to_json
import json
import requests

apify_client = ApifyClient('Kc4Q23smfu8GAcZzf9LWD4kNq')
# https://api.apify.com/v2/acts/clockworks~tiktok-comments-scraper/runs?token=Kc4Q23smfu8GAcZzf9LWD4kNq
# https://api.apify.com/v2/acts/clockworks~tiktok-comments-scraper?token=Kc4Q23smfu8GAcZzf9LWD4kNq
# https://github.com/googleapis/python-videointelligence/blob/HEAD/samples/analyze/analyze.py
# https://github.com/naver/splade
# https://github.com/beir-cellar/beir
# https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9

# !community war!
# https://www.tiktok.com/@missmaamshe/video/7198675731440766250
# https://www.tiktok.com/@briancamcab/video/7074979828386073861

# json_path = '/Users/jordanharris/Code/PycharmProjects/adsense/data/dataset_tiktok-comments-scraper_2023-03-06_06-57-20-881.json'
# html_path = '/Users/jordanharris/Code/PycharmProjects/adsense/data/HP_jk_rowling_homophobia.html'
html_path = '/Users/jordanharris/Code/PycharmProjects/adsense/data/HP_TRANSFOBIA_ALL.html'

with open(html_path, "r") as html_file:
    html = html_file.read()
    json_data = html_to_json.convert(html)

all_tt_c = json_data['html'][0]['body'][0]['div'][1]['div'][1]['div'][1]['div'][0]['div'][1]['div'][0]['div'][2]['div'][1]['div']
first_comment_date = json_data['html'][0]['body'][0]['div'][1]['div'][1]['div'][1]['div'][0]['div'][1]['div'][0]['div'][2]['div'][1]['div'][0]['div'][0]['div'][0]['p'][1]['span'][0]['_value']
first_comment_likes = json_data['html'][0]['body'][0]['div'][1]['div'][1]['div'][1]['div'][0]['div'][1]['div'][0]['div'][2]['div'][1]['div'][0]['div'][0]['div'][0]['p'][1]['div'][0]['span'][0]['_value']
first_nest = json_data['html'][0]['body'][0]['div'][1]['div'][1]['div'][1]['div'][0]['div'][1]['div'][0]['div'][2]['div'][1]['div'][2]['div'][0]['div'][0]['p'][0]['span'][0]['_value']

replys = []
for c in all_tt_c:
    uname_at = c['div'][0]['div'][0]['a'][0]['_attributes']['href']
    uname =    c['div'][0]['div'][0]['a'][0]['span'][0]['_value']
    c_date =   c['div'][0]['div'][0]['p'][1]['span'][0]['_value']
    c_likes =  c['div'][0]['div'][0]['p'][1]['div'][0]['span'][0]['_value']
    try:
        comment =  c['div'][0]['div'][0]['p'][0]['span'][0]['_value']
    except:
        user_ref = c['div'][0]['div'][0]['p'][0]['a'][0]['_value']
        unsure = c['div'][0]['div'][0]['p'][0]['a'][0]['_attributes']['href']
        comment = [user_ref, unsure]

    if len(all_tt_c[all_tt_c.index(c)]['div']) > 1:
        nests = all_tt_c[all_tt_c.index(c)]['div']
        nest = []
        for n in nests:
            if 'Action' not in n['div'][0]['_attributes']['class'][0]:
                try:
                    n_uname_at = n['div'][0]['a'][0]['_attributes']['href']
                    n_uname = n['div'][0]['a'][0]['_attributes']['href']
                    n_c_date = n['div'][0]['p'][1]['span'][0]['_value']
                    n_c_likes = n['div'][0]['p'][1]['div'][0]['span'][0]['_value']
                except:
                    n_uname_at = nests[0]['a'][0]['_attributes']['href']
                    n_uname = n['div'][0]['div'][0]['a'][0]['_attributes']['href']
                    n_c_date = n['div'][0]['div'][0]['p'][1]['span'][0]['_value']
                    n_c_likes = n['div'][0]['div'][0]['p'][1]['div'][0]['span'][0]['_value']

                try:
                    n_comment = n['div'][0]['div'][0]['p'][0]['span'][0]['_value']
                except:
                    try:
                        n_comment = n['div'][0]['p'][0]['span'][0]['_value']
                    except:
                        user_ref = c['div'][0]['div'][0]['p'][0]['a'][0]['_value']
                        unsure = c['div'][0]['div'][0]['p'][0]['a'][0]['_attributes']['href']
                        n_comment = [user_ref, unsure]

                nest.append([n_uname_at, n_uname, n_c_date, n_c_likes, n_comment])
            else:
                nest = None
        replys.append([uname_at, uname, comment, c_date, c_likes, nest])
print(replys)

# product review request lexipro: 7 comments
# https://www.tiktok.com/@sugarplumgoth/video/7206625893601643818
# 2312:Scifi book review
# https://www.tiktok.com/@speculativesandbox/video/7184073443938159914
# Cultural Information
# https://www.tiktok.com/@dez2fly/video/7180731993196399918
# human suggestion
# https://www.tiktok.com/@theonlyladyblu/video/7184146962571136302
# Music sample id w/ confirmation
# https://www.tiktok.com/@danzas.delmundo/video/7194505798163369222
# food/recipe ingredient ID w/ confirmation
# https://www.tiktok.com/@beautycooks00/video/7194985366045216043
# /https://www.tiktok.com/@victoriaada4/video/7200112509964094762?browserMode=1
# Date confirmation
# https://www.tiktok.com/@uniiqu3music/video/7197106545832873262
# light id
# https://in.tiktok.com/@franchesca_leigh/video/7021324815248149806
# book id
# https://www.tiktok.com/@kacimerriwetherhawkins/video/7116591639036071214
# PersonId
# https://www.tiktok.com/@realamaarae/video/7117672378599099653?lang=en
# Artist id:
# https://www.tiktok.com/t/ZTRWa7Myh/
# request for product feature review
# https://www.tiktok.com/@jhene.armani/video/7202405938198760750?_r=1&_t=8aPoLl3e33L
# digital product/tooling id
# https://www.tiktok.com/@georgedooley_/video/7190822605614615810?_r=1&_t=8aPpP8OPP1K
# product review & cultural references
# https://www.tiktok.com/@georgedooley_/video/7190822605614615810?_r=1&_t=8aPpP8OPP1K
# request for more info cultural/lexical
# https://www.tiktok.com/@hakunamatiti/video/7201564920524213550?browserMode=1
# song id
# https://www.tiktok.com/@liatvv/video/7200065491145084166?_r=1&_t=8aPopE5tN0t
# https://www.tiktok.com/@georgeprker/video/7203070777552276741?_r=1&_t=8aPopE5tN0t
# https://www.tiktok.com/@jakemarcelo12/video/7203356478831447302?_r=1&_t=8aPpLwUebxm
# https://www.tiktok.com/@lockin4999999/video/7172313266910154026?_r=1&_t=8aPpvNyYw71
# digital product (video game id)
# https://www.tiktok.com/@thurdculturekid/video/7185188565033274629?_r=1&_t=8aPpZQiIgow
# ?????
# https://in.tiktok.com/@franchesca_leigh/video/7112492754197810478
