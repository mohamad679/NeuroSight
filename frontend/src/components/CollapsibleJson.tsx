import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/cjs/styles/hljs";
import type { JsonValue } from "@/lib/types";

interface CollapsibleJsonProps {
  title: string;
  value: JsonValue;
  defaultOpen?: boolean;
}

export function CollapsibleJson({ title, value, defaultOpen = false }: CollapsibleJsonProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const Icon = isOpen ? ChevronDown : ChevronRight;

  return (
    <div className="rounded-lg border border-borderSubtle bg-bgElevated">
      <button
        type="button"
        onClick={() => setIsOpen((next) => !next)}
        className="flex w-full items-center justify-between px-3 py-3 text-left text-[13px] font-semibold text-textPrimary"
      >
        <span>{title}</span>
        <Icon className="h-4 w-4 text-textSecondary" />
      </button>
      {isOpen ? (
        <div className="border-t border-borderSubtle">
          <SyntaxHighlighter language="json" style={atomOneDark} className="syntax-compact">
            {JSON.stringify(value, null, 2)}
          </SyntaxHighlighter>
        </div>
      ) : null}
    </div>
  );
}
