# David Kaplan
""" External imports """
import argparse

"""
Initilizes argument parser
"""
def init_parser():
    parser = argparse.ArgumentParser(description='Codechef Solutions Downloader')
    
    #problem_code
    parser.add_argument(
        '-pc', 
        type=str, 
        help='The unique problem code', 
        required=True) 

    #page    
    parser.add_argument(
        '-p', 
        type=int, 
        default=-1, 
        help='Number of pages to download')

    #language    
    parser.add_argument(
        '-l', 
        nargs='+', 
        type=str, 
        default=['All'], 
        help='Solutions with the specified languages (C, JAVA, PYTH)')

    #status_code
    parser.add_argument(
        '-sc', 
        nargs='+', 
        type=str, 
        default=['AC'], 
        help='Solutions with specific status (AC, WA, TLE)')
    
    return parser
