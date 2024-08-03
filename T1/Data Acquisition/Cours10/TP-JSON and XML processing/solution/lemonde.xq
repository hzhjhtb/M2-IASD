declare variable $keyword external;

<results>{
let $d := doc('https://www.lemonde.fr/rss/en_continu.xml')
for $i in $d//item
where contains(lower-case($i/title), lower-case($keyword))
return
  <result date="{$i/pubDate}">
    <title>{$i/title/text()}</title>
    <summary>{$i/description/text()}</summary>
  </result>
}</results>
