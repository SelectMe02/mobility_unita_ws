#!/usr/bin/env python3
"""Extract signal-controlled MGeo stop-line nodes and their adjacent links."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def records(document, group):
    """Accept MGeo sets stored as an idx-keyed object or a record array."""
    if isinstance(document, dict) and group in document:
        document = document[group]
    if isinstance(document, dict):
        items = document.items()
    elif isinstance(document, list):
        items = ((None, item) for item in document)
    else:
        raise ValueError('{} must be an object or array'.format(group))
    for key, item in items:
        if not isinstance(item, dict):
            raise ValueError('{} entry {} is not an object'.format(group, key))
        idx = item.get('idx', key)
        if idx is None or str(idx) == '':
            raise ValueError('{} entry is missing idx'.format(group))
        if key is not None and 'idx' in item and str(key) != str(idx):
            raise ValueError('{} key/idx mismatch: {}'.format(group, key))
        yield str(idx), item


def extract(node_document, link_document):
    from_links = defaultdict(list)
    to_links = defaultdict(list)
    link_ids = set()
    for idx, link in records(link_document, 'links'):
        if idx in link_ids:
            raise ValueError('duplicate link idx: {}'.format(idx))
        link_ids.add(idx)
        for field, target in (('from_node_idx', from_links),
                              ('to_node_idx', to_links)):
            node_idx = link.get(field)
            if node_idx is None or str(node_idx) == '':
                raise ValueError('link {} is missing {}'.format(idx, field))
            target[str(node_idx)].append(idx)

    stoplines = []
    node_ids = set()
    for idx, node in records(node_document, 'nodes'):
        if idx in node_ids:
            raise ValueError('duplicate node idx: {}'.format(idx))
        node_ids.add(idx)
        if node.get('on_stop_line') is not True:
            continue
        light_id = node.get('traffic_light_id')
        if not isinstance(light_id, str) or not light_id.strip():
            continue
        point = node.get('point')
        if (not isinstance(point, list) or len(point) < 2
                or any(not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in point[:2])):
            raise ValueError('stop-line node {} has invalid point'.format(idx))
        stoplines.append({
            'idx': idx, 'x': point[0], 'y': point[1],
            'traffic_light_id': light_id,
            'from_link_idxs': sorted(from_links[idx]),
            'to_link_idxs': sorted(to_links[idx]),
        })
    stoplines.sort(key=lambda item: item['idx'])
    return stoplines


def statistics(stoplines):
    return {
        'stopline_nodes': len(stoplines),
        'from_link_references': sum(len(s['from_link_idxs']) for s in stoplines),
        'to_link_references': sum(len(s['to_link_idxs']) for s in stoplines),
        'nodes_with_from_links': sum(bool(s['from_link_idxs']) for s in stoplines),
        'nodes_with_to_links': sum(bool(s['to_link_idxs']) for s in stoplines),
        'nodes_with_both': sum(bool(s['from_link_idxs']) and bool(s['to_link_idxs'])
                               for s in stoplines),
        'nodes_with_neither': sum(not s['from_link_idxs'] and not s['to_link_idxs']
                                  for s in stoplines),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('node_set', type=Path)
    parser.add_argument('link_set', type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        'config/stoplines.json')
    args = parser.parse_args()
    with args.node_set.open(encoding='utf-8') as stream:
        nodes = json.load(stream)
    with args.link_set.open(encoding='utf-8') as stream:
        links = json.load(stream)
    stoplines = extract(nodes, links)
    if not stoplines:
        raise ValueError('no signal-controlled stop-line nodes found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stoplines, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    print('output: {}'.format(args.output))
    for name, value in statistics(stoplines).items():
        print('{}: {}'.format(name, value))


if __name__ == '__main__':
    main()
