# this code implements this taking inspiration from jina.ai 
pattern_heading = r"(?:^(?:[#*=-]{1,6}|\w[^\r\n]{0,199}\r?\n[-=]+|<h[1-6][^>]{0,99}>)[^\r\n]{1,200}(?:</h[1-6]>)?(?:\r?\n|$))"
pattern_items = r"(?:(?:^|\r?\n)[ \t]{0,3}(?:[-*+•]|\d{1,3}\.|\w\.|\[[ xX]\])[\t]+[^\r\n]{1,200}" \
                r"(?:(?:\r?\n[ \t]{2,5}(?:[-*+•]|\d{1,3}\.|\w\.|\[[ xX]\])[ \t]+[^\r\n]{1,200}){0,5 }" \
                r"(?: \r?\n[ \t]{4,7}(?:[-*+•]|\d{1,3}\.|\w\.|\[[ xX]\])[ \t]+[^\r\n]{1,200}){0,5})?)"

patten_blcoks = r"(?:(?:^>(?:>|\s{2,}){0,2}[^\r\n]{0,200}\r?\n?){1,10})"
pattern_codes = r"(?:(?:^|\r?\n)?(?:```|~~~)(?:\w{0,20})?\r?\n[\s\S]{0,1000}?(?:```|~~~)\r?\n?)" \
                r"(?:^|\r?\n)(?: {4}|\t)[^\r\n]{0,200}(?:\r?\n(?: {4}|\t)[^\r\n]{0,200}){0,20}\r?\n?" \
                r"(?:<pre>(?:<code>)?[\s\S]{0,1000}?(?:</code>)?</pre>)"
pattern_tables = r"(?:(?:^|\r?\n)(?:\|[^\r\n]{0,200}\|(?:\r?\n\|[-:]{1,200}\|){0,1}(?:\r?\\|[^\r\n]{0,200}\|){0,20}<table>[\s\S]{0,2000}?</table>))"
pattern_rules = r"(?:^(?:[-*_]{3,}\s*$|<hr\s*/?>))"
patten_sentences = r"(?:[^\r\n]{1,300}(?:[.!?...]|\.{3}|[\u2026\u2047-\u2049]|[😀-🙏🏻])(?:\s+|\r?\n|\Z))"
pattern_quoted_text = r"(?:\"\"\"[^\r\n]{0,300}\"\"\"" \
                    r"|['\"`][^\r\n]{0,300}['\"`]" \
                    r"|\([^\r\n()]{0,200}(?:\([^\r\n()]{0,200}\)[^\r\n()]{0,200}){0,5}\)" \
                    r"|\[[^\r\n[\]]{0,200}(?:\[[^\r\n\[\]]]{0,200}\][^\r\n\[\]]{0,200}){0,5}\]" \
                    r"|\$[^\r\n$]{0,100}\$" \
                    r"|`[^\r\n]{0,100}`)"
pattern_paragraphs = r"(?:(?:<p>)?(?:(?!\r?\n\r?\n|\Z).){1,1000}(?:</p >)?(?=\r?\n\r?\n|\Z))"
pattern_stand_alone_links = r"(?:(?:<[a-zA-Z:][^>]{0,99}>)?[^\r\n]{1,200}(?:<[a-zA-Z]+>)?(?:\r?\n|$))"
pattern_html_tags = r"(?:<[a-zA-Z][^>]{0,99}(?:>[\s\S]{0,1000}?</[a-zA-Z]+>|\s*/>))"
pattern_latex_tags = r"(?:(?:\$\$[\s\S]{0,500}?\$\$)|(?:\$[^$\r\n]{0,100}\$))"
pattern_emoji = r"(?:[\u2600-\u26FF\U0001F300-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0])"
pattern_all = r"(?:[^\r\n]{1,200})"

combined_pattern = r"(" \
    + pattern_heading + r"|" \
    + pattern_items + r"|" \
    + patten_blcoks + r"|" \
    + pattern_codes + r"|" \
    + pattern_tables + r"|" \
    + pattern_rules + r"|" \
    + patten_sentences + r"|" \
    + pattern_quoted_text + r"|" \
    + pattern_paragraphs + r"|" \
    + pattern_stand_alone_links + r"|" \
    + pattern_html_tags + r"|" \
    + pattern_latex_tags + r"|" \
    + pattern_emoji + r"|" \
    + pattern_all + r")"
