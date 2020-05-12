import pandas as pd

# def sic_as_num(df):
#     def first_sic(sics):
#         tmp = str(sics).split('\r\n')
#         tmp = [t.strip() for t in tmp]
#         tmp = [re.sub(',', '', t) for t in tmp]
#         tmp = [np.float(x) / 100 if x != 1 else -1 for x in tmp]
#         return tmp[0] if len(tmp) > 0 else -1

#     df['FirstSicCodeAsNum'] = df.SicCodes.map(first_sic)
#     return df

# def hot_enc_toplevel_sic_sector(df):
#     # See https://en.wikipedia.org/wiki/Standard_Industrial_Classification
    # mappings = {
    #     'Agriculture': ('01', '09'),
    #     'Mining': ('10', '14'),
    #     'Construction': ('15', '17'),
    #     'Manufacturing': ('20', '39'),
    #     'UtilityServices': ('40', '49'),
    #     'WholesaleTrade': ('50', '51'),
    #     'RetailTrade': ('52', '59'),
    #     'Financials': ('60', '69'),
    #     'Services': ('70', '90'),
    #     'PublicAdministration': ('91', '97'),
    #     'Nonclassifiable': ('98', '99')
    # }

#     df['CleanedUpSic'] = df.SicCodes.map(clean_sic)

#   def contains_range(sector, list_of_sics):
#         for sic in list_of_sics:
#             from_code, to_code = mappings[sector]
#             if from_code <= sic[:2] <= to_code:
#                 return 1.
#         return 0.

#     for key in mappings.keys():
#         df['Sector{}'.format(key)] = df.CleanedUpSic.map(lambda x: contains_range(key, x))

#     df.drop('CleanedUpSic', axis=1, inplace=True)
#     return df

# def drop_no_sicdata(df):
#     df = df.dropna(axis=0, subset=['SicCodes'])
#     return df

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

def explode_sectors(df, save_file=True, output_filename='data/ukgov-gpg-full-sectors.csv'):
    df = codes_to(df, str)
    df = parse_codes(df)
    df = df.explode('SicCodes')
    df = encode_missing_values(df)
    df = codes_to(df, int)
    codes = load_codes()
    df = pd.merge(df, codes, on=['SicCodes'])
    section_dummies = pd.get_dummies(df['Section'], prefix="Sect", prefix_sep="")
    df = pd.concat([df, section_dummies], axis = 1)
    df.drop('SicCodes', axis=1, inplace=True)
    if save_file: df.to_csv(output_filename, index=False)
    return df

def main():
    df = pd.read_csv('data/ukgov-gpg-full.csv')
    df = explode_sectors(df, save_file=True)

if __name__ == "__main__":
    main()
