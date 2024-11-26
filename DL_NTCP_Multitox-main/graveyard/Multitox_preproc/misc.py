import os




def get_all_combinations(toxicity, timepoint, spacer="_"):
    """
    This function generates column names for all of the combinations of toxicities and timepoints
    e.g. input ["Xerostomia","Taste"] and ["M06", "M12"] will return:
    ["Xerostomia_M06", "Xerostomia_M12", "Taste_M06", "Taste_M12"]
    """
    # get all the possible pairs for toxicity and timepoint
    pairs = [(toxicity[i], timepoint[j]) for i in range(len(toxicity))
            for j in range(len(timepoint))]
    
    # join each pair into a string, with the 'spacer' in-between
    keys = [spacer.join([i,j]) for i,j in pairs]

    return keys


