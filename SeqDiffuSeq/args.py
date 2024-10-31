from args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
import json

args = vars(create_argparser().parse_args())
print(type(args))

with open('args_config.json', 'w') as fp:
    json.dump(args, fp)