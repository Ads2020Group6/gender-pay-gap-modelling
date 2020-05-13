import pandas as pd
import numpy as np

def augment(data):
    data['MalePercent'] = (data['MaleLowerQuartile'] +
                         data['MaleLowerMiddleQuartile'] + data['MaleUpperMiddleQuartile'] +
                         data['MaleTopQuartile']) *.25
    data['FemalePercent'] = (data['FemaleLowerQuartile'] + data['FemaleLowerMiddleQuartile'] +
                           data['FemaleUpperMiddleQuartile'] + data['FemaleTopQuartile']
                           )*.25
    data['WorkforceGenderSkew'] = data['MalePercent'] - data['FemalePercent']

    data['PercMaleWorkforceInTopQuartile'] =  data['MaleTopQuartile'] / data['MalePercent'] * .25
    data['PercMaleWorkforceInUpperMiddleQuartile'] =  data['MaleUpperMiddleQuartile'] / data['MalePercent'] * .25
    data['PercMaleWorkforceInLowerMiddleQuartile'] =  data['MaleLowerMiddleQuartile'] / data['MalePercent'] * .25
    data['PercMaleWorkforceInLowerQuartile'] =  data['MaleLowerQuartile'] / data['MalePercent'] * .25

    data['PercFemaleWorkforceInTopQuartile'] =  data['FemaleTopQuartile'] /data['FemalePercent'] * .25
    data['PercFemaleWorkforceInUpperMiddleQuartile'] =  data['FemaleUpperMiddleQuartile'] / data['FemalePercent'] * .25
    data['PercFemaleWorkforceInLowerMiddleQuartile'] =  data['FemaleLowerMiddleQuartile'] / data['FemalePercent'] * .25
    data['PercFemaleWorkforceInLowerQuartile'] =  data['FemaleLowerQuartile'] / data['FemalePercent'] * .25

    data['RepresentationInTopQuartileSkew'] = data['PercMaleWorkforceInTopQuartile'] - data['PercFemaleWorkforceInTopQuartile']
    data['RepresentationInUpperMiddleQuartileSkew'] = data['PercMaleWorkforceInUpperMiddleQuartile'] - data['PercFemaleWorkforceInUpperMiddleQuartile']
    data['RepresentationInLowerMiddleQuartileSkew'] = data['PercMaleWorkforceInLowerMiddleQuartile'] - data['PercFemaleWorkforceInLowerMiddleQuartile']
    data['RepresentationInLowerQuartileSkew'] = data['PercMaleWorkforceInLowerQuartile'] - data['PercFemaleWorkforceInLowerQuartile']

    data['BonusGenderSkew'] = data['MaleBonusPercent'] - data['FemaleBonusPercent']
    return data

def main():
    df = pd.read_csv('data/ukgov-gpg-all-cleaned.csv')
    df = augment(df)
    df.to_csv('data/ukgov-gpg-all-clean-with-features.csv', index=False)

if __name__ == "__main__":
    main()
