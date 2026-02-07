# Security Policy

## 🔐 Reporting Security Issues

If you discover a security vulnerability in **openinfer**, please **do not open a public issue**.

Instead, report it privately by:

* emailing the project maintainer (add your preferred contact here), or
* using GitHub’s private security advisory feature if enabled.

When reporting, please include as much detail as possible:

* description of the vulnerability
* affected components (HAL, backend, runtime, allocator, etc.)
* steps to reproduce
* proof-of-concept if available
* potential impact
* platform / device / driver versions involved

Responsible disclosure is appreciated and helps keep users safe ❤️

---

## 🧭 Scope

Security concerns may include (but aren’t limited to):

* memory safety bugs
* buffer overflows / underflows
* use-after-free
* race conditions
* privilege escalation
* sandbox escapes
* unsafe device driver interactions
* kernel launch vulnerabilities
* malformed model input crashes
* denial-of-service via crafted workloads
* supply-chain risks

Even issues that “just” cause crashes are worth reporting.

---

## ⏳ Disclosure Process

Once a report is received:

1. I’ll acknowledge it as quickly as possible.
2. I’ll investigate and assess severity.
3. A fix will be developed privately when appropriate.
4. A security advisory and patch will be published once mitigation is ready.

Timelines may vary depending on complexity and hardware access.

---

## 🧪 Security Expectations for Contributors

When submitting changes, especially to HAL layers, allocators, or device backends, contributors are expected to:

* avoid unsafe memory practices where possible
* add tests for security-sensitive code
* document assumptions about device behavior
* avoid undefined behavior
* validate inputs from models or runtimes
* keep experimental drivers gated behind feature flags
* not weaken existing safety checks without discussion

---

## 🚫 Unsupported / Experimental Backends

Some backends or drivers may be marked as **experimental**.

These:

* may not receive immediate security fixes
* should not be used in production
* are provided for research or testing only

Such components will be clearly labeled in documentation.

---

## 🛡️ Safe Harbor

I support good-faith security research.

If you:

* make a genuine effort to avoid harming users
* report issues privately
* do not exploit data or systems beyond proof-of-concept
* allow reasonable time to fix issues

Then you are welcome to investigate and responsibly disclose vulnerabilities.

---

## 📌 Final Notes

OpenInfer is an evolving systems project. Security, correctness, and stability are core goals, especially as new hardware backends and execution models are added.

Thank you to everyone who helps make the project safer 🙏
