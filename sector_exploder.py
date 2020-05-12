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
#     mappings = {
#         'Agriculture': ('01', '09'),
#         'Mining': ('10', '14'),
#         'Construction': ('15', '17'),
#         'Manufacturing': ('20', '39'),
#         'UtilityServices': ('40', '49'),
#         'WholesaleTrade': ('50', '51'),
#         'RetailTrade': ('52', '59'),
#         'Financials': ('60', '69'),
#         'Services': ('70', '90'),
#         'PublicAdministration': ('91', '97'),
#         'Nonclassifiable': ('98', '99')
#     }

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

def ensure_codes_string(df):
    df.SicCodes = df.SicCodes.astype(str)
    return df

def strip_and_split(codes):
    return codes.replace('\r\n','').split(',')

def parse_codes(df):
    df.SicCodes = df.SicCodes.apply(strip_and_split)
    return df

def explode_sectors(df):
    df = ensure_codes_string(df)
    df = parse_codes(df)
    df = df.explode('SicCodes')    
    # df = drop_no_sicdata(df)
    # df = numerical_company_size(df)
    # df = one_hot_enc_company_size(df)
    # df = hot_enc_toplevel_sic_sector(df)
    # df = sic_as_num(df)
    # df.drop('SicCodes', axis=1, inplace=True)
    return df

# def main():
df = pd.read_csv('data/ukgov-gpg-full.csv')
df = explode_sectors(df)

# if __name__ == "__main__":
    # main()
