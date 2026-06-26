import { TrendingDown, TrendingUp } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface MetricCardProps {
  /** Small muted label above the number. */
  label: string;
  value: string | number;
  unit?: string;
  /** Corner outline badge with a trend arrow. */
  trend?: { value: string; up: boolean };
  /** Bold footer line (e.g. a short trend statement). */
  footerStrong?: React.ReactNode;
  /** Muted footer subtitle. */
  footerMuted?: React.ReactNode;
  valueClassName?: string;
}

/**
 * KPI tile in the shadcn dashboard-01 / Kiranism visual language:
 * description → big tabular number → corner trend badge → footer trend lines.
 * The faint top gradient comes from the grid wrapper's
 * `*:data-[slot=card]` classes, so this stays a plain Card.
 */
export function MetricCard({
  label,
  value,
  unit,
  trend,
  footerStrong,
  footerMuted,
  valueClassName,
}: MetricCardProps) {
  const TrendIcon = trend?.up ? TrendingUp : TrendingDown;
  return (
    <Card className="@container/card">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle
          className={cn(
            "text-2xl font-semibold tabular-nums @[250px]/card:text-3xl",
            valueClassName,
          )}
        >
          {value}
          {unit && (
            <span className="ml-1 text-base font-normal text-muted-foreground">
              {unit}
            </span>
          )}
        </CardTitle>
        {trend && (
          <CardAction>
            <Badge variant="outline">
              <TrendIcon />
              {trend.value}
            </Badge>
          </CardAction>
        )}
      </CardHeader>
      {(footerStrong || footerMuted) && (
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          {footerStrong && (
            <div className="line-clamp-1 flex items-center gap-2 font-medium">
              {footerStrong}
              {trend && <TrendIcon className="size-4" />}
            </div>
          )}
          {footerMuted && (
            <div className="text-muted-foreground">{footerMuted}</div>
          )}
        </CardFooter>
      )}
    </Card>
  );
}
