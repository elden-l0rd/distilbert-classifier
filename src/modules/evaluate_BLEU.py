import pandas as pd
import sacrebleu

list = ['data/results/translated_mt.xlsx',
        'data/results/translated_bn.xlsx',
        'data/results/translated_af.xlsx',
        'data/results/translated_kn.xlsx',
        ]

df = pd.read_excel('data/external/mitre-classified.xlsx')

with open('data/results/bleu_scores_translated.txt', 'w') as results:
    for file in list:
        df_t = pd.read_excel(file)
        bleu_scores = []
        
        for original, translated in zip(df['NameDesc'], df_t['NameDesc']):
            bleu_score = sacrebleu.corpus_bleu([translated], [[original]]).score
            bleu_scores.append(bleu_score)

        bleu_scores_series = pd.Series(bleu_scores)
        highest_bleu = bleu_scores_series.max()
        lowest_bleu = bleu_scores_series.min()
        median_bleu = bleu_scores_series.median()
        mean_bleu = bleu_scores_series.mean()
        # print(f"File: {file}")
        # print(f"Highest BLEU Score: {highest_bleu}")
        # print(f"Lowest BLEU Score: {lowest_bleu}")
        # print(f"Median BLEU Score: {median_bleu}")
        # print(f"Mean BLEU Score: {mean_bleu}")
        # print("=========================================\n")
        results.write(f"File: {file}\n")
        results.write(f"Highest BLEU Score: {highest_bleu}\n")
        results.write(f"Lowest BLEU Score: {lowest_bleu}\n")
        results.write(f"Median BLEU Score: {median_bleu}\n")
        results.write(f"Mean BLEU Score: {mean_bleu}\n")
        results.write("=========================================\n\n")
