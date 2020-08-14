#!/usr/bin/env python
import xnat
import argparse
import re


def get_files(connection, project, subject, session, scan):
    xnat_project = connection.projects[project]
    xnat_subject = xnat_project.subjects[subject]
    xnat_experiment = xnat_subject.experiments[session]
    xnat_scan = xnat_experiment.scans[scan]
    files = xnat_scan.files.values()
    return files


def filter_files(xnat_files, regex):
    filtered_files = []
    regex = re.compile(regex)
    for file in xnat_files:
        found = regex.match(file.name)
        if found:
            filtered_files.append(file)
    return filtered_files


def main():
    parser = argparse.ArgumentParser(description='Prints all files from a certain scan.')
    parser.add_argument('--xnathost', type=unicode, required=True, help='xnat host name')
    parser.add_argument('--project', type=unicode, required=True, help='Project id')
    parser.add_argument('--subject', type=unicode, required=True, help='subject')
    parser.add_argument('--session', type=unicode, required=True, help='session')
    parser.add_argument('--scan', type=unicode, required=True, help='scan')
    parser.add_argument('--filter', type=unicode, required=False, default='.*', help='regex filter for file names')
    args = parser.parse_args()
    with xnat.connect(args.xnathost) as connection:
        xnat_files = get_files(connection, args.project, args.subject, args.session, args.scan)
        xnat_files = filter_files(xnat_files, args.filter)
        for file in xnat_files:
            print('{}'.format(file.name))


if __name__ == '__main__':
    main()
