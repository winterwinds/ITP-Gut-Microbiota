import json

class FeatureManager:
    def __init__(self):
        self.features = {
            "species" : [
                    "g__Bacteroides;s__Bacteroides xylanisolvens",
                    "g__Bacteroides;s__Bacteroides sp. 3_1_23",
                    "g__Bacteroides;s__Bacteroides ovatus",
                    "g__Bacteroides;s__Bacteroides sp. D2",
                    "g__Ruminococcus;s__Ruminococcus faecis",
                    "g__Parabacteroides;s__Parabacteroides gordonii",
                    "g__Bacteroides;s__Bacteroides sp. 2_2_4",
                    "g__Turicibacter;s__Turicibacter sanguinis",
                    "g__Desulfovibrio;s__Desulfovibrio sp. An276",
                    "g__Megamonas;s__Megamonas rupellensis"
                    ],

            "alpha_diversity" : ["chao1", "dominance", "shannon_10"],

            "kegg_level3" : [
                    "ko00280", "ko00785", "ko04068", "ko01501", "ko04013", 
                    "ko04211", "ko01503", "ko05016", "ko00020", "ko00440"
                    ],

            "module" : [
                    "M00532", "M00131", "M00741", "M00231", "M00009",
                    "M00036", "M00011", "M00628", "M00373", "M00064"
                    ],

            "clinical" : [
                    "gender", "age", "duration_month",  
                    "PLT", "whobleeding", 
                    "acute_chronic"
                ]
        }

    def get_all_features(self):
        all_features = []
        for feature_list in self.features.values():
            all_features.extend(feature_list)
        return all_features
    
    def save_to_file(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.features, f, indent=4)

    def load_from_file(self, file_path):
        with open(file_path, 'r') as f:
            self.features = json.load(f)