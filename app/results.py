"""A result preserves text even when a later model stage fails."""
from dataclasses import dataclass, field

@dataclass
class Result:
    sections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    devices: list[dict] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)

    def merge(self, other):
        for name in ("sections", "warnings", "devices", "details"):
            getattr(self, name).extend(getattr(other, name))

    def text(self, source):
        header = f"Source: {source}\nStatus: {'PARTIAL / REVIEW REQUIRED' if self.warnings else 'completed'}\n"
        if self.warnings:
            header += "\nProcessing notes:\n" + "\n".join(f"- {w}" for w in self.warnings) + "\n"
        return header + "\n" + "\n\n".join(self.sections) + "\n"
