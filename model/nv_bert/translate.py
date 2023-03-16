import pandas as pd
import sqlite3
import re
import os
import os.path as osp


def get_token_types(input_source):
    # print('input_source:', input_src)

    token_types = ''

    for ele in re.findall('<n>.*</n>', input_source)[0].split(' '):
        token_types += ' nl'

    for ele in re.findall('<c>.*</c>', input_source)[0].split(' '):
        token_types += ' template'

    token_types += ' table table'

    for ele in re.findall('<col>.*</col>', input_source)[0].split(' '):
        token_types += ' col'

    for ele in re.findall('<val>.*</val>', input_source)[0].split(' '):
        token_types += ' value'

    token_types += ' table'

    token_types = token_types.strip()
    return token_types


def fix_chart_template(chart_template=None):
    query_template = \
        'mark [T] ' \
        'data [D] ' \
        'encoding x [X] y aggregate [AggFunction] [Y] ' \
        'color [Z] transform filter [F] ' \
        'group [G] ' \
        'bin [B] ' \
        'sort [S] ' \
        'topk [K]'

    if chart_template != None:
        try:
            query_template = query_template.replace('[T]', chart_template['chart'])
        except:
            raise ValueError('Error at settings of chart type!')

        try:
            if 'sorting_options' in chart_template and chart_template['sorting_options'] != None:
                order_xy = '[O]'
                if 'axis' in chart_template['sorting_options']:
                    if chart_template['sorting_options']['axis'].lower() == 'x':
                        order_xy = '[X]'
                    elif chart_template['sorting_options']['axis'].lower() == 'y':
                        order_xy = '[Y]'
                    else:
                        order_xy = '[O]'

                order_type = 'ASC'
                if 'type' in chart_template['sorting_options']:
                    if chart_template['sorting_options']['type'].lower() == 'desc':
                        order_type = 'DESC'
                    elif chart_template['sorting_options']['type'].lower() == 'asc':
                        order_type = 'ASC'
                    else:
                        raise ValueError('Unknown order by settings, the order-type must be "desc", or "asc"')
                query_template = query_template.replace('sort [S]', 'sort  ' +order_xy +'  ' +order_type)
        except:
            raise ValueError('Error at settings of sorting!')

        return query_template
    else:
        return query_template
