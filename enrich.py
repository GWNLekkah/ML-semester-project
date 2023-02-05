import json
import collections
import datetime
from dateutil.parser import parse


def get_new_data(keys):
    original_keys = keys.copy()
    from jira import JIRA
    APACHE_JIRA_SERVER = 'https://issues.apache.org/jira/'
    jira = JIRA(APACHE_JIRA_SERVER)
    fields = ['fixVersion', 'affectedVersion', 'environment']
    search = []
    while len(keys) > 100:
        key_str = ','.join(keys[0:100])
        keys = keys[100:]
        search.extend(jira.search_issues(f'key in ({key_str})', maxResults=1000, fields=fields))
    key_str = ','.join(keys)
    search.extend(jira.search_issues(f'key in ({key_str})', maxResults=1000, fields=fields))
    #search = jira.search_issues(f'key in ({",".join(keys)})', maxResults=3000, fields=fields)
    #fmt = '%a, %d %b %Y %H:%M:%S %z'
    #has_resolution = lambda i: hasattr(i.fields, 'resolutiondate') and i.fields.resolutiondate is not None
    #return {
    #    key: (parse(issue.fields.resolutiondate) - parse(issue.fields.created)).total_seconds()
    #    for key, issue in zip(original_keys, search) if has_resolution(issue)
    #}
    has_env = 0
    data = {}
    for key, issue in zip(original_keys, search):
        #print(dir(issue.fields))
        data[key] = {
            'num_fix_versions': len(issue.fields.fixVersions) if hasattr(issue.fields, 'fixVersions') else 0,
            'num_affected_versions': len(issue.fields.versions) if hasattr(issue.fields, 'versions') else 0
        }
        has_env += bool(hasattr(issue.fields, 'environment') and (issue.fields.environment is not None and issue.fields.environment))
    print('env', has_env)
    return data


def main():
    with open('new_data.json') as file:
        data = json.load(file)
    keys = [i['key'] for i in data]
    new_info = get_new_data(keys)
    for issue in data:
        issue |= new_info[issue['key']]
    with open('features.json', 'w') as file:
        json.dump(data, file, indent=4)



main()
