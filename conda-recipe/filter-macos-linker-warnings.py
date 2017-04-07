"""
When using gcc on newer versions of macOS/Xcode, the linker spits out a TON of deprecation warnings, which look like this:

/var/tmp//ccpBvJgm.s:44575:11: warning: section "__textcoal_nt" is deprecated
        .section __TEXT,__textcoal_nt,coalesced,pure_instructions
                 ^      ~~~~~~~~~~~~~
/var/tmp//ccpBvJgm.s:44575:11: note: change section name to "__text"
        .section __TEXT,__textcoal_nt,coalesced,pure_instructions
                 ^      ~~~~~~~~~~~~~
                 
(And similar.)

The volume of such messages makes it hard to scan the rest of the compiler output.
This script will filter out the noise.

Call it like this, using bash process substitution to capture only stderr:

$ my-build-command 2> >(python filter-macos-linker-warnings.py)

Examples:

$ make -j4 2> >(python filter-macos-linker-warnings.py)
$ conda build --python=3.5 --numpy=1.11 vigra 2> >(python build-utils/filter-macos-linker-warnings.py)

Unfortunately, the messages can get garbled before they even get processed by this script, due to parallel builds (e.g. "make -j4").
In those cases, you'll get occasional messages like this:

/var/tmp//ccpBvJgm.s:44229/var/tmp//cc1NpUCT.s::1132869: :warning: 11section "__textcoal_nt" is deprecated:
warning: section "__const_coal" is deprecated
                ..sseeccttiioonn  ____DTAETXAT,,____ctoenxsttc_ocaola_ln,tc,ocaolaelsecsecde
d , p u r e _ i n s t r u c t i o n^s
          ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
^      ~~~/var/tmp//cc1NpUCT.s~:~32869~:~11~: ~note: ~change section name to "__const"~
~    ~

There's not much we can do about those, but at least they are few in number.  The vast majority of the warning noise is correctly filtered out.
"""

import sys

for line in sys.stdin:
    # If line is a 'noisy' warning, don't print it or the following two lines.
    if ('warning: section' in line and 'is deprecated' in line
    or 'note: change section name to' in line):
        next(sys.stdin)
        next(sys.stdin)
    else:
        sys.stderr.write(line)
        sys.stderr.flush()
