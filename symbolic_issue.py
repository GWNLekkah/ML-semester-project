{
	"key": "string",			# Unique Issue Identifier
	"priority": "integer"		# 0 <= p <= 3; Issue priority from highest to lower; ordinal
	"resolution": [				# resolution (e.g. final judgement). Categorical variable, one-hot encoded. May be null
		1,							# done/implemented/resolved/fixed 
		0,							# invalid issue (e.g. duplicate)
		0,							# won't fix 
		0							# fix later 
	],
	"status": [					# status. Categorical variable, one-hot encoded 
		1,							# Resolved 
		0,							# Closed 
		0							# Other (open / in-progress)
	],
	"issuetype": [				# Type of the issue. Categorical, one-hot encoded 
		1,							# New Feature 
		0,							# Improvement 
		0,							# Task 
		0,							# Sub-task 
		0,							# Bug 
		0,							# Test 
		0							# Wish 
	],
	"labels": [					# Labels attached to the issue. Categorical and multivalued.
		0,							# Coordinated Coding Effort (e.g. coding sprint)
        0,							# Component Element Name 
        0,							# Component Name 
        0,							# Connector Data Name  
        0,							# Connector Name 
        0,							# Feature 
        0,							# Internal Protocol 
        0,							# Miscellaneous 
        0,							# Pattern Name 
        0,							# Quality Attribute Name 
        0,							# Release Name (e.g. 4.0)
        0,							# Tax-Better (eg. improve)
        0,							# Tax-Depend 
        0,							# Tax-Easy 
        0,							# Tax-Hard 
        0,							# Tax-Problem 
        0,							# Tax-Programming Activity 
        0							# Technology Name 
	],
	"resolution_time": 1.0		# Time for creation to resolving in seconds. May be null
	"n_components": 0,			# number of components affected by the issue 
	"n_labels": 0,				# Total number of labels attached to the issue
	"n_comments": 0,			# Amount of comments left on the issue 
	"n_attachments": 0,			# Amount of files attached to the issue 
	"n_votes": 0,				# Amount of votes for the issue 
	"n_watches": 0,				# Amount of people watching the issue 
	"n_issuelinks": 0,			# Number of links to other issues 
	"n_subtasks": 0,			# Number of sub-tasks related to this issue 
	"parent": 0,				# Boolean indicating whether this issue has a parent issue 
	"len_summary": 0,			# Length of the issue summary in characters 
	"len_description": 0,		# Length of the issue description in charachters 
	"len_comments": 0,			# Cumulative length of all comments left on the issue 
	"avg_comment": 0,			# Average length of a comment left on the issue 
}