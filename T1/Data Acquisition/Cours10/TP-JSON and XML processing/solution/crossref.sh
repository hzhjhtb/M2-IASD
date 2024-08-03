#!/bin/sh

curl --get 'https://api.crossref.org/journals?rows=1000' --data-urlencode query="$1" |
  jq '
[ .message.items[] |
  { title : .title,
    pub_per_year: (
      [ .breakdowns."dois-by-issued-year"[][1] ] |
      if . == [] then 0 else add / length | round end
    )
  }
] | sort_by(.pub_per_year)
'
