#!/usr/bin/env python3

import bz2
import re
import sys
import xml.sax

class WikiHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.mode = ''

    def startElement(self, tag, attributes):
        if self.mode == 'skip':
            return

        if tag == 'title' or tag == 'text':
            self.mode = tag

    def endElement(self, tag):
        if self.mode == 'skip' and tag != 'page':
            return
        self.mode = ''

        if tag == 'title':
            print(self.content+"\t", end="")
        elif tag == 'text':
            # Normalize spaces
            self.content = re.sub(r"\s+", " ", self.content)

            # Imperfect cleaning of links
            self.content = re.sub(r"\[\[[^\]]*\|([^|]*?)]]", r"\1", self.content)
            self.content = re.sub(r"\[\[(.*?)]]", r"\1", self.content)

            # Imperfect removal of templates
            while True:
                oldcontent = self.content
                self.content = re.sub(r"{[{|][^{]*?[}|]}", "", self.content)
                if oldcontent == self.content:
                    break
            print(self.content)
        self.content = ''

    def characters(self, text):
        if self.mode == 'title' and ":" in text:
            # Special page, ignore
            self.mode = 'skip'
        elif self.mode == 'title' or self.mode == 'text':
            self.content += text

if len(sys.argv) != 1:
    print("Usage: %s < file.xml.bz2"%(sys.argv[0]), file=sys.stderr)
    sys.exit(1)

with bz2.open(sys.stdin.buffer) as stream:
    parser = xml.sax.make_parser()
    parser.setContentHandler(WikiHandler())
    parser.parse(stream)
