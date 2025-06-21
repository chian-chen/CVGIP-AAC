python3 CAAC.py \
    --datasets kodak \
    --output './logs' \
    --prediction_methods EDP GAP MED DIFF \
    --context_settings BASE MIXED\
    --context_type_features BASE

python3 CAAC2.py \
    --datasets kodak \
    --output './logs' \
    --prediction_methods EDP GAP MED DIFF \
    --context_settings BASE MIXED\
    --context_type_features BASE

python3 CAAC3.py \
    --datasets kodak \
    --output './logs' \
    --prediction_methods EDP GAP MED DIFF \
    --context_settings BASE MIXED\
    --context_type_features BASE


   

# python3 CAAC.py \
#     --datasets datas Kodak hdr \
#     --output './logs' \
#     --prediction_methods EDP GAP MED DIFF \
#     --context_settings BASE \
#     --context_type_features BASE
#     --visualization 

# def parse_args():
#     parser = argparse.ArgumentParser(description="CAAC Settings")
#     parser.add_argument('--datasets', nargs='+', default=['datas'],
#                         help='Dataset names or paths to process')
#     parser.add_argument('--output', type=str, default='./output',
#                         help='Output folder for processed images')
#     parser.add_argument('--prediction_methods', nargs='+', help='Prediction method to use')
#     parser.add_argument('--context_settings', nargs='+', help='Context settings for the prediction methods')
#     parser.add_argument('--context_type_features', nargs='+', help='Context features for the prediction methods')
#     parser.add_argument('--visualization', action='store_true', help='Enable visualization of the prediction process')
#     parser.add_argument('--context_features_num', type=int, default=4, help='Number of context features to use')
#     return parser.parse_args()