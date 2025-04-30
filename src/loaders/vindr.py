import pandas as pd
import os, ast

class VindrDataframeLoader(object):

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.birads_mapping = {
            'bi-rads_1': '1',
            'bi-rads_2': '2',
            'bi-rads_3': '0',
            'bi-rads_4': '0',
            'bi-rads_5': '0'
        }
        
        self.lesions_mapping = {
            'no_finding': '0',
            'mass': '1',
            'suspicious_calcifications': '2'
        }

    def format_char(self, char):
        return char.lower().replace(' ', '_')

    def format_category_list(self, category_list):
        return [self.format_char(category) for category in category_list]

    def contains_all_classes(self, category_list, class_list):
        return any(cls in category_list for cls in class_list)

    def replace_categories(self, df, column, target_categories):
        def replace_if_present(categories):
            for target in target_categories:
                if target in categories:
                    return target
            return categories

        df[column] = df[column].apply(
            lambda x: replace_if_present(x) if isinstance(x, list) else x)

    def _construct_image_path(self, row: pd.Series) -> str:
        return os.path.join(self.data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')

    def __call__(self, split='training'):
        df_find = pd.read_csv(os.path.join(
            self.data_dir, 'finding_annotations.csv'))
        df_find['finding_categories'] = df_find['finding_categories'].apply(
            ast.literal_eval)
        df_find['finding_categories'] = df_find['finding_categories'].apply(
            self.format_category_list)
        df_find['breast_birads'] = df_find['breast_birads'].apply(
            self.format_char)
        df_find['breast_birads'] = df_find['breast_birads'].replace(
            self.birads_mapping)
        df_find['breast_birads'] = df_find['breast_birads'].str.extract(r'(\d+)')
        df_find.drop_duplicates(subset='image_id', keep=False, inplace=True)
        df_find = df_find[df_find['split'] == split]
        df_find['absolute_path'] = df_find.apply(self._construct_image_path, axis=1)
        return df_find
    
    def __call__(self, split='training'):
        df_find = pd.read_csv(os.path.join(self.data_dir, 'finding_annotations.csv'))

        df_find['finding_categories'] = df_find['finding_categories'].apply(ast.literal_eval)
        df_find['finding_categories'] = df_find['finding_categories'].apply(self.format_category_list)
        df_find['breast_birads'] = df_find['breast_birads'].apply(self.format_char)
        df_find['breast_birads'] = df_find['breast_birads'].replace(self.birads_mapping)
        df_find.drop_duplicates(subset='image_id', keep=False, inplace=True)
        df_find = df_find[df_find['split'] == split]

        # Replace and filter finding categories
        target_categories = ['mass', 'no_finding', 'suspicious_calcifications']
        self.replace_categories(df_find, 'finding_categories', target_categories)
        df_find = df_find[df_find['finding_categories'].isin(target_categories)]
        df_find['finding_categories'] = df_find['finding_categories'].replace(self.lesions_mapping)

        # Add absolute path
        df_find['absolute_path'] = df_find.apply(self._construct_image_path, axis=1)

        return df_find