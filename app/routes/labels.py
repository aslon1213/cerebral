from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import CurrentUser, LabelAccess, LabelRepoDep
from app.core.response import Response, error_responses, ok
from app.repo.base import Page, SortOrder
from app.repo.labels import Label, LabelCreate, LabelResponse, LabelSort, LabelUpdate

# Every route here needs a bearer token, and answers 403 for a label owned by
# somebody else; the rest is documented per route.
router = APIRouter(
    prefix="/labels", tags=["labels"], responses=error_responses(401, 403)
)


@router.post("", status_code=status.HTTP_201_CREATED, responses=error_responses(409))
async def create_label(
    body: LabelCreate, current_user: CurrentUser, labels: LabelRepoDep
) -> Response[LabelResponse]:
    label = Label(**body.model_dump(), created_by=current_user.id)
    created = await labels.create(label)
    return ok(LabelResponse.model_validate(created))


@router.get("")
async def list_labels(
    current_user: CurrentUser,
    labels: LabelRepoDep,
    q: Annotated[str | None, Query(max_length=100)] = None,
    sort_by: LabelSort = LabelSort.NAME,
    order: SortOrder = SortOrder.ASC,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[LabelResponse]]:
    items, total = await labels.list(
        current_user.id,
        q=q,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return ok(
        Page[LabelResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


@router.get("/{label_id}", responses=error_responses(404))
async def get_label(
    label_id: LabelAccess, current_user: CurrentUser, labels: LabelRepoDep
) -> Response[LabelResponse]:
    label = await labels.get(label_id, current_user.id)
    if label is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Label not found")
    return ok(LabelResponse.model_validate(label))


@router.patch("/{label_id}", responses=error_responses(404, 409))
async def update_label(
    label_id: LabelAccess,
    body: LabelUpdate,
    current_user: CurrentUser,
    labels: LabelRepoDep,
) -> Response[LabelResponse]:
    label = await labels.update(
        label_id, current_user.id, body.model_dump(exclude_unset=True)
    )
    if label is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Label not found")
    return ok(LabelResponse.model_validate(label))


@router.delete("/{label_id}", responses=error_responses(404))
async def delete_label(
    label_id: LabelAccess, current_user: CurrentUser, labels: LabelRepoDep
) -> Response[None]:
    if not await labels.delete(label_id, current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Label not found")
    return ok(None)
