Deviation from preregistration invocation (recorded before campaign start):
the installed validator resolves candidate TOMLs at parents[2]/config of the
installed module (/usr/local/lib/python3.12/config), a path the image does not
populate (the same files exist at /opt/project/config). Remedy: read-only bind
mount of the pinned checkout's config/ at the searched path. Mounted bytes are
hash-identical to both the checked-in TOMLs at the source commit and the
image's /opt/project/config copies (see sibling hash files). No managed
environment value changed. Code fix (search /opt/project/config) queued as
follow-up.
