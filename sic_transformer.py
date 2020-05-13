import pandas as pd
import numpy as np

def codes_to(df, typ):
    df.SicCodes = df.SicCodes.astype(typ)
    return df

def strip_and_split(codes):
    return codes.replace('\r\n','').split(',')

def parse_codes(df):
    df.SicCodes = df.SicCodes.apply(strip_and_split)
    return df

def encode_missing_values(df):
    df.SicCodes.replace(to_replace='nan', value='2', inplace=True)
    return df

def load_codes():
    codes = pd.read_csv('data/sic_codes.csv')
    codes.rename(columns={
            "sic_code": "SicCodes",
            "section": "Section",
            "section_description": "SectionDesc"
        }, inplace=True)
    codes.drop(['sic_version', 'sic_description'], axis='columns', inplace=True)
    codes = codes_to(codes, int)
    return codes

# def get_codes_dict():
#     codes = load_codes()
#     code_to_section = {}
#     for i, sic_code in enumerate(codes.SicCodes):
#         row = codes.iloc[i]
#         code_to_section[row.SicCodes] = row.Section
#     return code_to_section

def get_unique_sections():
    return pd.unique(load_codes().Section)

def build_empty_dummies(df, sections):
    zeroes = np.zeros((df.shape[0], len(sections)))
    return pd.DataFrame(zeroes, columns=sections)

def explode_sectors(df, save_file=False, output_filename='data/ukgov-gpg-full-sectors.csv'):
    df = codes_to(df, str)
    df = parse_codes(df)
    df = df.explode('SicCodes')
    df = encode_missing_values(df)
    df = codes_to(df, int)
    codes = load_codes()
    df = pd.merge(df, codes, on=['SicCodes'])
    section_dummies = pd.get_dummies(df['Section'], prefix="Sect", prefix_sep="")
    df = pd.concat([df, section_dummies], axis = 1)
    df.drop(['SicCodes','Section','SectionDesc'], axis=1, inplace=True)
    if save_file: df.to_csv(output_filename, index=False)
    return df

def split_sectors(df):
    df = codes_to(df, str)
    df = parse_codes(df)
    sections = get_unique_sections()
    print(sections)
    dummies = build_empty_dummies(df, sections)
    print(dummies.shape)
    # df = pd.merge(df, load_codes(), on=['SicCodes'])
    # for i, sic_code in enumerate(df.SicCodes):
    #     indices = dummies.columns.get_indexer()
    # df = encode_missing_values(df)
    return df

def main():
    df = pd.read_csv('data/ukgov-gpg-full.csv')
    # df = explode_sectors(df, save_file=True)
    return split_sectors(df)

if __name__ == "__main__":
    df = main()
