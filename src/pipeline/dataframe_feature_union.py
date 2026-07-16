from sklearn.pipeline import FeatureUnion


class DataFrameFeatureUnion(FeatureUnion):
    """
    A `FeatureUnion` that restores the `get_feature_names` behaviour removed
    from scikit-learn (since 1.0, dropped in later releases).

    It delegates to the custom `get_feature_names` implemented by each member
    `DataFrameColumnTransformer` and prefixes every returned feature with the
    member name, i.e. ``{name}__{feature}`` - matching the naming scheme the
    original scikit-learn implementation produced.
    """

    def get_feature_names(self):
        feature_names = []
        for name, transformer in self.transformer_list:
            for feature in transformer.get_feature_names():
                feature_names.append(f"{name}__{feature}")
        return feature_names
