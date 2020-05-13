import pandas as pd
import numpy as np

def drop_sic_codes_na(df):
    return df.dropna(subset=['SicCodes'])

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

def build_code_to_section_dict():
    codes = load_codes()
    code_to_section = {}
    for i, sic_code in enumerate(codes.SicCodes):
        row = codes.iloc[i]
        code_to_section[row.SicCodes] = row.Section
    code_to_section[1] = "Unknown"
    return code_to_section

def get_unique_sections():
    return pd.unique(load_codes().Section)

def build_empty_dummies(df, sections):
    zeroes = np.zeros((df.shape[0], len(sections)))
    return pd.DataFrame(zeroes, columns=sections, index=df.index)

def generate_dummies(df, sections):
    dummies = build_empty_dummies(df, sections)
    code_to_section = build_code_to_section_dict()
    for i, sic_codes in enumerate(df.SicCodes):
        sections = [code_to_section[int(code)] for code in sic_codes]
        indices = np.unique(dummies.columns.get_indexer(sections))
        dummies.iloc[i, indices] = 1
    return dummies.add_prefix('Sect')

def explode_sectors(df):
    print("Exploding Industry Sections")
    df = codes_to(df, str)
    df = parse_codes(df)
    df = df.explode('SicCodes')
    df = encode_missing_values(df)
    df = codes_to(df, int)
    codes = load_codes()
    df = pd.merge(df, codes, on=['SicCodes'])
    section_dummies = pd.get_dummies(df['Section'], prefix="Sect", prefix_sep="")
    df = pd.concat([df, section_dummies], axis = 1)
    df.drop(['Section','SectionDesc'], axis=1, inplace=True)
    return df

def split_sectors(df):
    print("Splitting Industry Sections")
    df = df.copy()
    df = drop_sic_codes_na(df)
    df = codes_to(df, str)
    df = parse_codes(df)
    sections = get_unique_sections()
    dummies = generate_dummies(df, sections)
    df = df.join(dummies)
    return df

def main():
    df = pd.read_csv('data/ukgov-gpg-full.csv')
    # explode_sectors(df)
    split_sectors(df)

if __name__ == "__main__":
    main()
